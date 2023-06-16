import torch
from torch import Tensor
from torch import nn
#from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from PIL import Image
import random
from collections import namedtuple
import torchvision.transforms as T
import torch.nn.functional as F
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
import numpy as np
class PatchEmbedding(nn.Module):
	def __init__(self, in_channels: int = 2, patch_size: int = 15, emb_size: int = 765, img_size: int = [60 ,135]):
		self.patch_size = patch_size
		super().__init__()
		self.projection = nn.Sequential(
			# using a conv layer instead of a linear one -> performance gains
			nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
			Rearrange('b e (h) (w) -> b (h w) e'),
		)
		self.cls_token = nn.Parameter(torch.randn(1 ,1, emb_size))
		self.positions = nn.Parameter \
			(torch.randn(((img_size[0] // patch_size ) *(img_size[1] // patch_size) ) +1, emb_size))

	def forward(self, x: Tensor) -> Tensor:

		b, _, _, _ = x.shape
		x = self.projection(x)
		cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
		# prepend the cls token to the input
		x = torch.cat([cls_tokens, x], dim=1)
		# add position embedding

		x += self.positions

		return x
class MultiHeadAttention(nn.Module):
	def __init__(self, emb_size: int = 765, num_heads: int = 5, dropout: float = 0):
		super().__init__()
		self.emb_size = emb_size
		self.num_heads = num_heads
		# fuse the queries, keys and values in one matrix
		self.qkv = nn.Linear(emb_size, emb_size * 3)
		self.att_drop = nn.Dropout(dropout)
		self.projection = nn.Linear(emb_size, emb_size)
		self.scaling = (self.emb_size // num_heads) ** -0.5


	def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
		# split keys, queries and values in num_heads
		qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
		queries, keys, values = qkv[0], qkv[1], qkv[2]
		# sum up over the last axis
		energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
		if mask is not None:
			fill_value = torch.finfo(torch.float32).min
			energy.mask_fill(~mask, fill_value)


		att = F.softmax(energy * self.scaling, dim=-1)
		att = self.att_drop(att)
		# sum up over the third axis
		out = torch.einsum('bhal, bhlv -> bhav ', att, values)
		out = rearrange(out, "b h n d -> b n (h d)")
		out = self.projection(out)
		return out


class ResidualAdd(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x, **kwargs):
		res = x
		x = self.fn(x, **kwargs)
		x += res
		return x


# In[9]:


class FeedForwardBlock(nn.Sequential):  # MLP
	def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
		super().__init__(
			nn.Linear(emb_size, expansion * emb_size),
			nn.GELU(),
			nn.Dropout(drop_p),
			nn.Linear(expansion * emb_size, emb_size),
		)


# In[10]:


class TransformerEncoderBlock(nn.Sequential):
	def __init__(self,
				 emb_size: int = 765,
				 drop_p: float = 0.,
				 forward_expansion: int = 4,
				 forward_drop_p: float = 0.,
				 ** kwargs):
		super().__init__(
			ResidualAdd(nn.Sequential(
				nn.LayerNorm(emb_size),
				MultiHeadAttention(emb_size, **kwargs),
				nn.Dropout(drop_p)
			)),
			ResidualAdd(nn.Sequential(
				nn.LayerNorm(emb_size),
				FeedForwardBlock(
					emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
				nn.Dropout(drop_p)
			)
			))

class TransformerEncoder(nn.Sequential):
	def __init__(self, depth: int = 3, **kwargs):
		super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


# In[12]:


class ViT(nn.Sequential):
	def __init__(self,
				 in_channels: int = 2,
				 patch_size: int = 15,
				 emb_size: int = 765,
				 img_size: int = [60, 135],
				 depth: int = 3,
				 args = None,
				 **kwargs):

		#parse args and get variables
		self.args = args
		self.HIDDEN_LAYER_1 = args.hidden_layer_1
		self.HIDDEN_LAYER_2 = args.hidden_layer_2
		self.HIDDEN_LAYER_3 = args.hidden_layer_3
		self.KERNEL_SIZE = args.kernel_size
		self.STRIDE = args.stride
		self.DROPOUT = args.dropout
		self.FRAMES = args.frames
		self.RESIZE_PIXELS = args.resize_pixels
		self.GRAYSCALE = args.grayscale

		# Settings for GRAYSCALE / RGB
		if self.GRAYSCALE == 0:
			self.resize = T.Compose([T.ToPILImage(),
								T.Resize((self.RESIZE_PIXELS), interpolation=Image.CUBIC),
								T.ToTensor()])

			nn_inputs = 3 * self.FRAMES  # number of channels for the nn
		else:
			self.resize = T.Compose([T.ToPILImage(),
								T.Resize((self.RESIZE_PIXELS), interpolation=Image.CUBIC),
								T.Grayscale(),
								T.ToTensor()])
			nn_inputs = self.FRAMES  # number of channels for the nn

		super().__init__(
			PatchEmbedding(in_channels, patch_size, emb_size, img_size),
			TransformerEncoder(depth, emb_size=emb_size, **kwargs),

		)



# Memory for Experience Replay
class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)  # if we haven't reached full capacity, we append a new transition
		self.memory[self.position] = Transition(*args)
		self.position = (
									self.position + 1) % self.capacity  # e.g if the capacity is 100, and our position is now 101, we don't append to

	# position 101 (impossible), but to position 1 (its remainder), overwriting old data

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


# Build CNN
class DQN(nn.Module):

	def __init__(self, h, w, outputs, args):
		super(DQN, self).__init__()

		#parse args and get variables
		self.args = args
		self.HIDDEN_LAYER_1 = args.hidden_layer_1
		self.HIDDEN_LAYER_2 = args.hidden_layer_2
		self.HIDDEN_LAYER_3 = args.hidden_layer_3
		self.KERNEL_SIZE = args.kernel_size
		self.STRIDE = args.stride
		self.DROPOUT = args.dropout
		self.FRAMES = args.frames
		self.RESIZE_PIXELS = args.resize_pixels
		self.GRAYSCALE = args.grayscale

		# Settings for GRAYSCALE / RGB
		if self.GRAYSCALE == 0:
			self.resize = T.Compose([T.ToPILImage(),
								T.Resize((self.RESIZE_PIXELS), interpolation=Image.CUBIC),
								T.ToTensor()])

			nn_inputs = 3 * self.FRAMES  # number of channels for the nn
		else:
			self.resize = T.Compose([T.ToPILImage(),
								T.Resize((self.RESIZE_PIXELS), interpolation=Image.CUBIC),
								T.Grayscale(),
								T.ToTensor()])
			nn_inputs = self.FRAMES  # number of channels for the nn

		self.conv1 = nn.Conv2d(nn_inputs, self.HIDDEN_LAYER_1, kernel_size=self.KERNEL_SIZE, stride=self.STRIDE)
		self.bn1 = nn.BatchNorm2d(self.HIDDEN_LAYER_1)
		self.conv2 = nn.Conv2d(self.HIDDEN_LAYER_1, self.HIDDEN_LAYER_2, kernel_size=self.KERNEL_SIZE, stride=self.STRIDE)
		self.bn2 = nn.BatchNorm2d(self.HIDDEN_LAYER_2)
		self.conv3 = nn.Conv2d(self.HIDDEN_LAYER_2, self.HIDDEN_LAYER_3, kernel_size=self.KERNEL_SIZE, stride=self.STRIDE)
		self.bn3 = nn.BatchNorm2d(self.HIDDEN_LAYER_3)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size=self.KERNEL_SIZE, stride=self.STRIDE):
			return (size - (kernel_size - 1) - 1) // stride + 1

		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw * convh * 32

		# print("인풋 사이즈",linear_input_size)
		self.head = nn.Linear(linear_input_size, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		y = self.head(x.view(x.size(0), -1))
		# print("info",x.shape,x.size(0),x.view(x.size(0), -1).shape,y.shape)
		return self.head(x.view(x.size(0), -1))

# In[19]:

# Build vision-transformer 3 depth
class DQN2(nn.Module):

	def __init__(self, h, w, outputs, args):
		super(DQN2, self).__init__()

		#parse args and get variables
		self.args = args
		self.HIDDEN_LAYER_1 = args.hidden_layer_1
		self.HIDDEN_LAYER_2 = args.hidden_layer_2
		self.HIDDEN_LAYER_3 = args.hidden_layer_3
		self.KERNEL_SIZE = args.kernel_size
		self.STRIDE = args.stride
		self.DROPOUT = args.dropout
		self.FRAMES = args.frames
		self.RESIZE_PIXELS = args.resize_pixels
		self.GRAYSCALE = args.grayscale

		# Settings for GRAYSCALE / RGB
		if self.GRAYSCALE == 0:
			self.resize = T.Compose([T.ToPILImage(),
								T.Resize((self.RESIZE_PIXELS), interpolation=Image.CUBIC),
								T.ToTensor()])

			nn_inputs = 3 * self.FRAMES  # number of channels for the nn
		else:
			self.resize = T.Compose([T.ToPILImage(),
								T.Resize((self.RESIZE_PIXELS), interpolation=Image.CUBIC),
								T.Grayscale(),
								T.ToTensor()])
			nn_inputs = self.FRAMES  # number of channels for the nn


		self.model = ViT(args = args)

		ex = torch.randn(1, self.FRAMES, h, w)
		linear_input_size = torch.flatten(self.model(ex)).shape[0]
		self.head = nn.Linear(linear_input_size, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = self.model(x)

		return self.head(x.view(x.size(0), -1))

