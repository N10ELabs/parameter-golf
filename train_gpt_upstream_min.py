from __future__ import annotations
_R='_use_new_zipfile_serialization'
_Q='pickle_protocol'
_P='momentum'
_O='TRAIN_SEQ_LEN'
_N='fineweb_train_*.bin'
_M='baseline_tensor_bytes'
_L='zlib'
_K='cpu'
_J='params'
_I='cuda'
_H='lr'
_G='int8_payload_bytes'
_F=.0
_E='0'
_D=1.
_C=False
_B=True
_A=None
import copy,glob,io,lzma,math,os,random,sys,time,uuid,zlib
from pathlib import Path
import numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch import Tensor,nn
from torch.nn.parallel import DistributedDataParallel as DDP
class Hyperparameters:data_path=os.environ.get('DATA_PATH','./data/datasets/fineweb10B_sp1024');train_files=os.path.join(data_path,_N);val_files=os.path.join(data_path,'fineweb_val_*.bin');tokenizer_path=os.environ.get('TOKENIZER_PATH','./data/tokenizers/fineweb_1024_bpe.model');run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));seed=int(os.environ.get('SEED',1337));val_batch_size=int(os.environ.get('VAL_BATCH_SIZE',524288));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',1000));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',200));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_iters=int(os.environ.get('WARMDOWN_ITERS',1200));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',524288));train_seq_len=int(os.environ.get(_O,1024));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',os.environ.get(_O,'1024')));eval_stride=int(os.environ.get('EVAL_STRIDE',_E));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',1.5));init_model_path=os.environ.get('INIT_MODEL_PATH','');disable_model_compile=bool(int(os.environ.get('DISABLE_MODEL_COMPILE',_E)));skip_prequant_eval_zero_iters=bool(int(os.environ.get('SKIP_PREQUANT_EVAL_ZERO_ITERS',_E)));qat_int4=bool(int(os.environ.get('QAT_INT4',_E)));qat_only_target_params=bool(int(os.environ.get('QAT_ONLY_TARGET_PARAMS',_E)));vocab_size=int(os.environ.get('VOCAB_SIZE',1024));num_layers=int(os.environ.get('NUM_LAYERS',9));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));model_dim=int(os.environ.get('MODEL_DIM',512));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=int(os.environ.get('MLP_MULT',2));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS','1')));rope_base=float(os.environ.get('ROPE_BASE',1e4));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.05));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.04));scalar_lr=float(os.environ.get('SCALAR_LR',.04));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.95));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_weight_decay=float(os.environ.get('MUON_WEIGHT_DECAY','0.0'));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.85));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',500));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',_F));model_compressor=os.environ.get('MODEL_COMPRESSOR',_L);model_compress_preset=int(os.environ.get('MODEL_COMPRESS_PRESET','9'));quant_pickle_protocol=int(os.environ.get('QUANT_PICKLE_PROTOCOL',_E));quant_use_new_zipfile_serialization=bool(int(os.environ.get('QUANT_USE_NEW_ZIPFILE_SERIALIZATION','1')));quant_load_weights_only=bool(int(os.environ.get('QUANT_LOAD_WEIGHTS_ONLY','1')))
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-07):
	D,E,F=3.4445,-4.775,2.0315;A=G.bfloat16();A/=A.norm()+eps;C=G.size(0)>G.size(1)
	if C:A=A.T
	for I in range(steps):B=A@A.T;H=E*B+F*B@B;A=D*A+H@A
	return A.T if C else A
class Muon(torch.optim.Optimizer):
	def __init__(A,params,lr,momentum,backend_steps,nesterov=_B):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov))
	@torch.no_grad()
	def step(self,closure=_A):
		J=closure;I='momentum_buffer';K=_A
		if J is not _A:
			with torch.enable_grad():K=J()
		F=dist.is_available()and dist.is_initialized();N=dist.get_world_size()if F else 1;O=dist.get_rank()if F else 0
		for D in self.param_groups:
			E=D[_J]
			if not E:continue
			P=D[_H];L=D[_P];Q=D['backend_steps'];R=D['nesterov'];S=sum(int(A.numel())for A in E);G=torch.zeros(S,device=E[0].device,dtype=torch.bfloat16);C=0
			for(T,B)in enumerate(E):
				if T%N==O and B.grad is not _A:
					A=B.grad;H=self.state[B]
					if I not in H:H[I]=torch.zeros_like(A)
					M=H[I];M.mul_(L).add_(A)
					if R:A=A.add(M,alpha=L)
					A=zeropower_via_newtonschulz5(A,steps=Q);A*=max(1,A.size(0)/A.size(1))**.5;G[C:C+B.numel()]=A.reshape(-1)
				C+=B.numel()
			if F:dist.all_reduce(G,op=dist.ReduceOp.SUM)
			C=0
			for B in E:A=G[C:C+B.numel()].view_as(B).to(dtype=B.dtype);B.add_(A,alpha=-P);C+=B.numel()
		return K
def build_sentencepiece_luts(sp,vocab_size,device):
	D=device;B=sp;G=int(B.vocab_size());E=max(G,vocab_size);F=np.zeros((E,),dtype=np.int16);H=np.zeros((E,),dtype=np.bool_);I=np.ones((E,),dtype=np.bool_)
	for A in range(G):
		if B.is_control(A)or B.is_unknown(A)or B.is_unused(A):continue
		I[A]=_C
		if B.is_byte(A):F[A]=1;continue
		C=B.id_to_piece(A)
		if C.startswith('▁'):H[A]=_B;C=C[1:]
		F[A]=len(C.encode('utf-8'))
	return torch.tensor(F,dtype=torch.int16,device=D),torch.tensor(H,dtype=torch.bool,device=D),torch.tensor(I,dtype=torch.bool,device=D)
def load_validation_tokens(pattern,seq_len):
	B=pattern;A=seq_len;C=[Path(A)for A in sorted(glob.glob(B))]
	if not C:raise FileNotFoundError(f"No files found for pattern: {B}")
	D=torch.cat([load_data_shard(A)for A in C]).contiguous()
	if A<=0:raise ValueError(f"Validation seq_len must be positive, got {A}")
	E=(D.numel()-1)//A*A
	if E<=0:raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={A}")
	return D[:E+1]
def eval_val(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut):
	d=is_boundary_token_lut;c=has_leading_space_lut;b=base_bytes_lut;a=grad_accum_steps;P=val_tokens;O=rank;N=model;I=device;H=world_size;D=args;A=D.eval_seq_len;V=D.val_batch_size//(H*a)
	if V<A:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={D.val_batch_size}, WORLD_SIZE={H}, GRAD_ACCUM_STEPS={a}, EVAL_SEQ_LEN={A}")
	Q=torch.zeros((),device=I,dtype=torch.float64);J=torch.zeros((),device=I,dtype=torch.float64);R=torch.zeros((),device=I,dtype=torch.float64);N.eval()
	with torch.inference_mode():
		if D.eval_stride<=0 and A==D.train_seq_len:
			e=V//A;f=(P.numel()-1)//A;t=f*O//H;g=f*(O+1)//H
			for h in range(t,g,e):
				u=min(h+e,g);v=h*A;w=u*A+1;K=P[v:w].to(device=I,dtype=torch.int64,non_blocking=_B);C=K[:-1].reshape(-1,A);B=K[1:].reshape(-1,A)
				with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):x=N(C,B).detach()
				i=float(B.numel());Q+=x.to(torch.float64)*i;J+=i;W=C.reshape(-1);L=B.reshape(-1);M=b[L].to(dtype=torch.int16);M+=(c[L]&~d[W]).to(dtype=torch.int16);R+=M.to(torch.float64).sum()
		else:
			if D.eval_stride<=0:raise ValueError('EVAL_STRIDE must be positive when EVAL_SEQ_LEN differs from TRAIN_SEQ_LEN')
			j=max(1,V//A);X=P.numel()-1;S=[];k=0
			for y in range(0,X,D.eval_stride):
				E=min(y+A,X);T=E-k
				if T>0:Y=max(0,E-A);S.append((Y,E,T))
				k=E
				if E>=X:break
			z=len(S)*O//H;l=len(S)*(O+1)//H
			for m in range(z,l,j):
				n=S[m:min(m+j,l)];Z=max(B-A for(A,B,C)in n);o=[];p=[];q=[]
				for(Y,E,T)in n:
					K=P[Y:E+1].to(device=I,dtype=torch.int64,non_blocking=_B);C=K[:-1];B=K[1:];U=int(C.numel())
					if U<Z:r=Z-U;C=F.pad(C,(0,r),value=0);B=F.pad(B,(0,r),value=0)
					G=torch.zeros(Z,device=I,dtype=torch.bool);G[U-T:U]=_B;o.append(C);p.append(B);q.append(G)
				C=torch.stack(o);B=torch.stack(p);G=torch.stack(q)
				with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):A0=N(C,_A)
				A1=F.cross_entropy(A0.float(),B.reshape(-1),reduction='none').view_as(B);Q+=A1[G].to(torch.float64).sum();J+=G.sum().to(torch.float64);W=C[G];L=B[G];M=b[L].to(dtype=torch.int16);M+=(c[L]&~d[W]).to(dtype=torch.int16);R+=M.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(Q,op=dist.ReduceOp.SUM);dist.all_reduce(J,op=dist.ReduceOp.SUM);dist.all_reduce(R,op=dist.ReduceOp.SUM)
	s=Q/J;A2=s.item()/math.log(2.);A3=J.item()/R.item();N.train();return float(s.item()),float(A2*A3)
CONTROL_TENSOR_NAME_PATTERNS=tuple(A for A in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights').split(',')if A)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS=tuple(A for A in os.environ.get('INT8_KEEP_FLOAT_FP32_NAME_PATTERNS',','.join(CONTROL_TENSOR_NAME_PATTERNS)).split(',')if A)
INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS=tuple(A for A in os.environ.get('INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS','').split(',')if A)
INT8_KEEP_FLOAT_MAX_NUMEL=65536
INT8_KEEP_FLOAT_STORE_DTYPE=torch.float16
INT8_PER_ROW_SCALE_DTYPE=torch.float16
INT8_CLIP_PERCENTILE=99.99984
INT8_CLIP_Q=INT8_CLIP_PERCENTILE/1e2
INT4_CLIP_PERCENTILE=float(os.environ.get('INT4_CLIP_PERCENTILE',str(INT8_CLIP_PERCENTILE)))
INT4_CLIP_Q=INT4_CLIP_PERCENTILE/1e2
INT4_GROUP_SIZE=int(os.environ.get('INT4_GROUP_SIZE',_E))
INT4_NAME_PATTERNS=tuple(A for A in os.environ.get('INT4_NAME_PATTERNS','mlp.fc.weight,mlp.proj.weight').split(',')if A)
def tensor_nbytes(t):return int(t.numel())*int(t.element_size())
def keep_float_tensor(name,t):
	if any(A in name for A in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):return t.float().contiguous()
	return t if t.dtype==INT8_KEEP_FLOAT_STORE_DTYPE else t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
def should_quantize_int4(name,t):return t.ndim==2 and any(A in name for A in INT4_NAME_PATTERNS)
def pack_int4_tensor(q):
	A=q.reshape(-1).to(torch.int16)
	if A.numel()&1:A=torch.cat((A,torch.zeros(1,dtype=torch.int16)))
	A=A&15;return(A[0::2]|A[1::2]<<4).to(torch.uint8).contiguous()
def unpack_int4_tensor(packed,numel):B=packed.reshape(-1).to(torch.int16);A=torch.stack((B&15,B>>4),dim=1).reshape(-1)[:numel];A=torch.where(A>=8,A-16,A);return A.to(torch.float32)
def reshape_int4_groups(t):
	if t.ndim!=2:raise ValueError('int4 grouping only supports 2D tensors')
	A=t.shape[1];B=A if INT4_GROUP_SIZE<=0 else min(INT4_GROUP_SIZE,A);C=math.ceil(A/B);D=C*B-A
	if D:t=F.pad(t,(0,D))
	return t.view(t.shape[0],C,B),A
def expand_int4_scales(scale,ref):
	B=ref;A=scale
	if A.ndim==1:return A.float().view(B.shape[0],1)
	C=math.ceil(B.shape[1]/A.shape[1]);return A.float().repeat_interleave(C,dim=1)[:,:B.shape[1]]
def fake_quantize_tensor_int4(t):
	if t.ndim!=2:return t
	A=t.float();B,E=reshape_int4_groups(A);C=torch.quantile(B.abs(),INT4_CLIP_Q,dim=2)if B.numel()else torch.empty(B.shape[:2],dtype=torch.float32,device=A.device);F=torch.maximum(torch.minimum(B,C[...,_A]),-C[...,_A]);D=(C/7.).clamp_min(_D/7.);G=(torch.clamp(torch.round(F/D[...,_A]),-7,7)*D[...,_A]).reshape(A.shape[0],-1)[:,:E];return(A+(G-A).detach()).to(dtype=t.dtype)
def quantize_float_tensor(t):
	A=t.float()
	if A.ndim==2:B=torch.quantile(A.abs(),INT8_CLIP_Q,dim=1)if A.numel()else torch.empty((A.shape[0],),dtype=torch.float32);E=torch.maximum(torch.minimum(A,B[:,_A]),-B[:,_A]);C=(B/127.).clamp_min(_D/127.);D=torch.clamp(torch.round(E/C[:,_A]),-127,127).to(torch.int8).contiguous();return D,C.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
	B=float(torch.quantile(A.abs().flatten(),INT8_CLIP_Q).item())if A.numel()else _F;C=torch.tensor(B/127. if B>0 else _D,dtype=torch.float32);D=torch.clamp(torch.round(torch.clamp(A,-B,B)/C),-127,127).to(torch.int8).contiguous();return D,C.to(dtype=INT8_PER_ROW_SCALE_DTYPE)
def quantize_float_tensor_int4(t):
	B=t.float()
	if B.ndim!=2:raise ValueError('int4 path only supports 2D tensors')
	A,E=reshape_int4_groups(B);C=torch.quantile(A.abs(),INT4_CLIP_Q,dim=2)if A.numel()else torch.empty(A.shape[:2],dtype=torch.float32);F=torch.maximum(torch.minimum(A,C[...,_A]),-C[...,_A]);D=(C/7.).clamp_min(_D/7.);G=torch.clamp(torch.round(F/D[...,_A]),-7,7).to(torch.int8).reshape(B.shape[0],-1)[:,:E].contiguous();return pack_int4_tensor(G),D.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
def quantize_state_dict_int8(state_dict):
	O='num_nonfloat_tensors';N='num_float_tensors';M='num_tensors';L='param_count';H={};I={};J={};K={};G={};A=dict.fromkeys((L,M,N,O,_M,_G),0)
	for(B,P)in state_dict.items():
		C=P.detach().to(_K).contiguous();A[L]+=int(C.numel());A[M]+=1;A[_M]+=tensor_nbytes(C)
		if not C.is_floating_point():A[O]+=1;G[B]=C;A[_G]+=tensor_nbytes(C);continue
		if any(A in B for A in INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS):D=keep_float_tensor(B,C);G[B]=D;A[_G]+=tensor_nbytes(D);continue
		if C.numel()<=INT8_KEEP_FLOAT_MAX_NUMEL:D=keep_float_tensor(B,C);G[B]=D;A[_G]+=tensor_nbytes(D);continue
		A[N]+=1
		if should_quantize_int4(B,C):E,F=quantize_float_tensor_int4(C);J[B]=E;K[B]=F;A[_G]+=tensor_nbytes(E)+tensor_nbytes(F)
		else:E,F=quantize_float_tensor(C);H[B]=E;I[B]=F;A[_G]+=tensor_nbytes(E)+tensor_nbytes(F)
	return{'q':H,'s':I,'q4':J,'s4':K,'p':G},A
def dequantize_state_dict_int8(obj,template_state):
	G=template_state;D=obj;E={}
	for(A,C)in D['q'].items():
		B=G[A];F=D['s'][A]
		if F.ndim:E[A]=(C.float()*F.float().view(C.shape[0],*[1]*(C.ndim-1))).to(dtype=B.dtype).contiguous()
		else:E[A]=(C.float()*float(F.item())).to(dtype=B.dtype).contiguous()
	for(A,C)in D.get('q4',{}).items():B=G[A];F=D['s4'][A];H=unpack_int4_tensor(C,B.numel()).view_as(B);E[A]=(H*expand_int4_scales(F,B)).to(dtype=B.dtype).contiguous()
	for(A,I)in D['p'].items():B=G[A];E[A]=I.detach().to(_K).to(dtype=B.dtype).contiguous()
	return E
def load_data_shard(file):
	H='<u2';G='<i4';A=file;D=256*np.dtype(G).itemsize;I=np.dtype(H).itemsize;B=np.fromfile(A,dtype=G,count=256)
	if B.size!=256 or int(B[0])!=20240520 or int(B[1])!=1:raise ValueError(f"Unexpected shard header for {A}")
	C=int(B[2]);E=D+C*I
	if A.stat().st_size!=E:raise ValueError(f"Shard size mismatch for {A}: expected {E} bytes")
	F=np.fromfile(A,dtype=H,count=C,offset=D)
	if F.size!=C:raise ValueError(f"Short read for {A}")
	return torch.from_numpy(F.astype(np.uint16,copy=_C))
class TokenStream:
	def __init__(A,pattern):
		B=pattern;A.files=[Path(A)for A in sorted(glob.glob(B))]
		if not A.files:raise FileNotFoundError(f"No files found for pattern: {B}")
		A.file_idx=0;A.tokens=load_data_shard(A.files[0]);A.pos=0
	def _advance_file(A):A.file_idx=(A.file_idx+1)%len(A.files);A.tokens=load_data_shard(A.files[A.file_idx]);A.pos=0
	def take(A,n):
		B=[];C=n
		while C>0:
			E=A.tokens.numel()-A.pos
			if E<=0:A._advance_file();continue
			D=min(C,E);B.append(A.tokens[A.pos:A.pos+D]);A.pos+=D;C-=D
		return B[0]if len(B)==1 else torch.cat(B)
class DistributedTokenLoader:
	def __init__(A,pattern,rank,world_size,device):A.rank=rank;A.world_size=world_size;A.device=device;A.stream=TokenStream(pattern)
	def next_batch(A,global_tokens,seq_len,grad_accum_steps):C=seq_len;F=global_tokens//(A.world_size*grad_accum_steps);B=F+1;G=A.stream.take(B*A.world_size);D=A.rank*B;E=G[D:D+B].to(dtype=torch.int64);H=E[:-1].reshape(-1,C);I=E[1:].reshape(-1,C);return H.to(A.device,non_blocking=_B),I.to(A.device,non_blocking=_B)
class RMSNorm(nn.Module):
	def __init__(A,eps=_A):super().__init__();A.eps=eps
	def forward(A,x):return F.rms_norm(x,(x.size(-1),),eps=A.eps)
class CastedLinear(nn.Linear):
	def forward(A,x):B=A.bias.to(x.dtype)if A.bias is not _A else _A;C=fake_quantize_tensor_int4(A.weight)if A.training and getattr(A,'_qat_int4',_C)else A.weight;return F.linear(x,C.to(x.dtype),B)
def restore_low_dim_params_to_fp32(module):
	with torch.no_grad():
		for(B,A)in module.named_parameters():
			if(A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS))and A.dtype!=torch.float32:A.data=A.data.float()
class Rotary(nn.Module):
	def __init__(A,dim,base=1e4,train_seq_len=1024):B=dim;super().__init__();A.dim=B;A.base=base;A.train_seq_len=train_seq_len;C=_D/base**(torch.arange(0,B,2,dtype=torch.float32)/B);A.register_buffer('inv_freq',C,persistent=_C);A._seq_len_cached=0;A._cos_cached=_A;A._sin_cached=_A
	def forward(A,seq_len,device,dtype):
		E=dtype;C=device;B=seq_len
		if A._cos_cached is _A or A._sin_cached is _A or A._seq_len_cached!=B or A._cos_cached.device!=C:
			if B>A.train_seq_len:G=B/A.train_seq_len;H=A.base*G**(A.dim/(A.dim-2));D=_D/H**(torch.arange(0,A.dim,2,dtype=torch.float32,device=C)/A.dim)
			else:D=A.inv_freq.to(C)
			I=torch.arange(B,device=C,dtype=D.dtype);F=torch.outer(I,D);A._cos_cached=F.cos()[_A,_A,:,:];A._sin_cached=F.sin()[_A,_A,:,:];A._seq_len_cached=B
		return A._cos_cached.to(dtype=E),A._sin_cached.to(dtype=E)
def apply_rotary_emb(x,cos,sin):A=x.size(-1)//2;B,C=x[...,:A],x[...,A:];return torch.cat((B*cos+C*sin,B*-sin+C*cos),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len=1024):
		D=num_kv_heads;C=num_heads;B=dim;super().__init__()
		if B%C!=0:raise ValueError('model_dim must be divisible by num_heads')
		if C%D!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
		A.num_heads=C;A.num_kv_heads=D;A.head_dim=B//C
		if A.head_dim%2!=0:raise ValueError('head_dim must be even for RoPE')
		E=A.num_kv_heads*A.head_dim;A.c_q=CastedLinear(B,B,bias=_C);A.c_k=CastedLinear(B,E,bias=_C);A.c_v=CastedLinear(B,E,bias=_C);A.proj=CastedLinear(B,B,bias=_C);A.proj._zero_init=_B;A.q_gain=nn.Parameter(torch.full((C,),qk_gain_init,dtype=torch.float32));A.rotary=Rotary(A.head_dim,base=rope_base,train_seq_len=train_seq_len)
	def forward(A,x):E,D,J=x.shape;B=A.c_q(x).reshape(E,D,A.num_heads,A.head_dim).transpose(1,2);C=A.c_k(x).reshape(E,D,A.num_kv_heads,A.head_dim).transpose(1,2);K=A.c_v(x).reshape(E,D,A.num_kv_heads,A.head_dim).transpose(1,2);B=F.rms_norm(B,(B.size(-1),));C=F.rms_norm(C,(C.size(-1),));H,I=A.rotary(D,x.device,B.dtype);B=apply_rotary_emb(B,H,I);C=apply_rotary_emb(C,H,I);B=B*A.q_gain.to(dtype=B.dtype)[_A,:,_A,_A];G=F.scaled_dot_product_attention(B,C,K,attn_mask=_A,is_causal=_B,enable_gqa=A.num_kv_heads!=A.num_heads);G=G.transpose(1,2).contiguous().reshape(E,D,J);return A.proj(G)
class MLP(nn.Module):
	def __init__(A,dim,mlp_mult):B=dim;super().__init__();C=mlp_mult*B;A.fc=CastedLinear(B,C,bias=_C);A.proj=CastedLinear(C,B,bias=_C);A.proj._zero_init=_B
	def forward(A,x):x=torch.relu(A.fc(x));return A.proj(x.square())
class Block(nn.Module):
	def __init__(A,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len=1024):B=dim;super().__init__();A.attn_norm=RMSNorm();A.mlp_norm=RMSNorm();A.attn=CausalSelfAttention(B,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len=train_seq_len);A.mlp=MLP(B,mlp_mult);A.attn_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.mlp_scale=nn.Parameter(torch.ones(B,dtype=torch.float32));A.resid_mix=nn.Parameter(torch.stack((torch.ones(B),torch.zeros(B))).float())
	def forward(A,x,x0):B=A.resid_mix.to(dtype=x.dtype);x=B[0][_A,_A,:]*x+B[1][_A,_A,:]*x0;C=A.attn(A.attn_norm(x));x=x+A.attn_scale.to(dtype=x.dtype)[_A,_A,:]*C;x=x+A.mlp_scale.to(dtype=x.dtype)[_A,_A,:]*A.mlp(A.mlp_norm(x));return x
class GPT(nn.Module):
	def __init__(A,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,tied_embed_init_std,logit_softcap,rope_base,qk_gain_init,train_seq_len=1024):
		F=tie_embeddings;E=vocab_size;D=logit_softcap;C=num_layers;B=model_dim;super().__init__()
		if D<=_F:raise ValueError(f"logit_softcap must be positive, got {D}")
		A.tie_embeddings=F;A.tied_embed_init_std=tied_embed_init_std;A.logit_softcap=D;A.tok_emb=nn.Embedding(E,B);A.num_encoder_layers=C//2;A.num_decoder_layers=C-A.num_encoder_layers;A.num_skip_weights=min(A.num_encoder_layers,A.num_decoder_layers);A.skip_weights=nn.Parameter(torch.ones(A.num_skip_weights,B,dtype=torch.float32));A.blocks=nn.ModuleList([Block(B,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len=train_seq_len)for A in range(C)]);A.final_norm=RMSNorm();A.lm_head=_A if F else CastedLinear(B,E,bias=_C)
		if A.lm_head is not _A:A.lm_head._zero_init=_B
		A._init_weights()
	def _init_weights(A):
		if A.tie_embeddings:nn.init.normal_(A.tok_emb.weight,mean=_F,std=A.tied_embed_init_std)
		for B in A.modules():
			if isinstance(B,nn.Linear)and getattr(B,'_zero_init',_C):nn.init.zeros_(B.weight)
	def forward(B,input_ids,target_ids=_A):
		E=target_ids;A=B.tok_emb(input_ids);A=F.rms_norm(A,(A.size(-1),));G=A;D=[]
		for C in range(B.num_encoder_layers):A=B.blocks[C](A,G);D.append(A)
		for C in range(B.num_decoder_layers):
			if D:A=A+B.skip_weights[C].to(dtype=A.dtype)[_A,_A,:]*D.pop()
			A=B.blocks[B.num_encoder_layers+C](A,G)
		A=B.final_norm(A).reshape(-1,A.size(-1))
		if B.tie_embeddings:H=F.linear(A,B.tok_emb.weight)
		else:
			if B.lm_head is _A:raise RuntimeError('lm_head is required when tie_embeddings=False')
			H=B.lm_head(A)
		I=B.logit_softcap*torch.tanh(H/B.logit_softcap)
		if E is _A:return I
		J=E.reshape(-1);return F.cross_entropy(I.float(),J,reduction='mean')
def save_quantized_state(obj,args):
	A=args;C=io.BytesIO();B={}
	if A.quant_pickle_protocol>0:B[_Q]=A.quant_pickle_protocol
	if not A.quant_use_new_zipfile_serialization:B[_R]=_C
	torch.save(obj,C,**B);D=C.getvalue()
	if A.model_compressor==_L:return zlib.compress(D,level=A.model_compress_preset)
	if A.model_compressor=='lzma':return lzma.compress(D,preset=A.model_compress_preset)
	raise ValueError(f"Unsupported MODEL_COMPRESSOR={A.model_compressor}")
def load_quantized_state(blob,args):
	A=args
	if A.model_compressor==_L:B=zlib.decompress(blob)
	elif A.model_compressor=='lzma':B=lzma.decompress(blob)
	else:raise ValueError(f"Unsupported MODEL_COMPRESSOR={A.model_compressor}")
	return torch.load(io.BytesIO(B),map_location=_K,weights_only=A.quant_load_weights_only)
def main():
	AC='final_model.pt';AB='WORLD_SIZE';p='final_model.int8.ptz';T='base_lr';global zeropower_via_newtonschulz5;A=Hyperparameters();zeropower_via_newtonschulz5=torch.compile(zeropower_via_newtonschulz5);I='RANK'in os.environ and AB in os.environ;P=int(os.environ.get('RANK',_E));F=int(os.environ.get(AB,'1'));q=int(os.environ.get('LOCAL_RANK',_E))
	if F<=0:raise ValueError(f"WORLD_SIZE must be positive, got {F}")
	if 8%F!=0:raise ValueError(f"WORLD_SIZE={F} must divide 8 so grad_accum_steps stays integral")
	E=8//F;r=_D/E
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	G=torch.device(_I,q);torch.cuda.set_device(G)
	if I:dist.init_process_group(backend='nccl',device_id=G);dist.barrier()
	U=P==0;torch.backends.cuda.matmul.allow_tf32=_B;torch.backends.cudnn.allow_tf32=_B;from torch.backends.cuda import enable_cudnn_sdp as AD,enable_flash_sdp as AE,enable_math_sdp as AF,enable_mem_efficient_sdp as AG;AD(_C);AE(_B);AG(_C);AF(_C);V=_A
	if U:os.makedirs('logs',exist_ok=_B);V=f"logs/{A.run_id}.txt";print(V)
	def B(msg,console=_B):
		if not U:return
		if console:print(msg)
		if V is not _A:
			with open(V,'a',encoding='utf-8')as A:print(msg,file=A)
	random.seed(A.seed);np.random.seed(A.seed);torch.manual_seed(A.seed);torch.cuda.manual_seed_all(A.seed)
	if not A.tokenizer_path.endswith('.model'):raise ValueError(f"Script only setup for SentencePiece .model file: {A.tokenizer_path}")
	Y=spm.SentencePieceProcessor(model_file=A.tokenizer_path)
	if int(Y.vocab_size())!=A.vocab_size:raise ValueError(f"VOCAB_SIZE={A.vocab_size} does not match tokenizer vocab_size={int(Y.vocab_size())}")
	s=Path(A.data_path).resolve();AH=len(list(s.glob(_N)));AI=max(A.train_seq_len,A.eval_seq_len);Z=load_validation_tokens(A.val_files,AI);t,u,v=build_sentencepiece_luts(Y,A.vocab_size,G);B(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={A.tokenizer_path}");B(f"train_loader:dataset:{s.name} train_shards:{AH}");B(f"val_loader:shards pattern={A.val_files} tokens:{Z.numel()-1}");C=GPT(vocab_size=A.vocab_size,num_layers=A.num_layers,model_dim=A.model_dim,num_heads=A.num_heads,num_kv_heads=A.num_kv_heads,mlp_mult=A.mlp_mult,tie_embeddings=A.tie_embeddings,tied_embed_init_std=A.tied_embed_init_std,logit_softcap=A.logit_softcap,rope_base=A.rope_base,qk_gain_init=A.qk_gain_init,train_seq_len=A.train_seq_len).to(G).bfloat16()
	for Q in C.modules():
		if isinstance(Q,CastedLinear):Q.float()
	restore_low_dim_params_to_fp32(C)
	if A.init_model_path:AJ=torch.load(A.init_model_path,map_location=_K);C.load_state_dict(AJ,strict=_B)
	w=0
	for(AK,Q)in C.named_modules():
		if isinstance(Q,CastedLinear):x=A.qat_int4 and any(A in f"{AK}.weight"for A in INT4_NAME_PATTERNS);Q._qat_int4=x;w+=int(x)
	if A.qat_only_target_params:
		for(AL,a)in C.named_parameters():a.requires_grad_(any(A in AL for A in INT4_NAME_PATTERNS))
	y=C if A.disable_model_compile else torch.compile(C,dynamic=_C,fullgraph=_B);J=DDP(y,device_ids=[q],broadcast_buffers=_C)if I else y;z=list(C.blocks.named_parameters());b=[A for(B,A)in z if A.requires_grad and A.ndim==2 and not any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS)];c=[A for(B,A)in z if A.requires_grad and(A.ndim<2 or any(A in B for A in CONTROL_TENSOR_NAME_PATTERNS))]
	if C.skip_weights.requires_grad and C.skip_weights.numel()>0:c.append(C.skip_weights)
	d=A.tied_embed_lr if A.tie_embeddings else A.embed_lr;H=[]
	if C.tok_emb.weight.requires_grad:H.append(torch.optim.Adam([{_J:[C.tok_emb.weight],_H:d,T:d}],betas=(A.beta1,A.beta2),eps=A.adam_eps,fused=_B))
	if C.lm_head is not _A and C.lm_head.weight.requires_grad:H.append(torch.optim.Adam([{_J:[C.lm_head.weight],_H:A.head_lr,T:A.head_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,fused=_B))
	K=_A
	if b:
		K=Muon(b,lr=A.matrix_lr,momentum=A.muon_momentum,backend_steps=A.muon_backend_steps)
		for M in K.param_groups:M[T]=A.matrix_lr
		H.append(K)
	if c:H.append(torch.optim.Adam([{_J:c,_H:A.scalar_lr,T:A.scalar_lr}],betas=(A.beta1,A.beta2),eps=A.adam_eps,fused=_B))
	AM=sum(A.numel()for A in C.parameters());AN=sum(A.numel()for A in C.parameters()if A.requires_grad);B(f"model_params:{AM}");B(f"trainable_params:{AN}");B(f"world_size:{F} grad_accum_steps:{E}");B('sdp_backends:cudnn=False flash=True mem_efficient=False math=False');B(f"attention_mode:gqa num_heads:{A.num_heads} num_kv_heads:{A.num_kv_heads}");B(f"tie_embeddings:{A.tie_embeddings} embed_lr:{d} head_lr:{A.head_lr if C.lm_head is not _A else _F} matrix_lr:{A.matrix_lr} scalar_lr:{A.scalar_lr} muon_weight_decay:{A.muon_weight_decay}");B(f"train_batch_tokens:{A.train_batch_tokens} train_seq_len:{A.train_seq_len} eval_seq_len:{A.eval_seq_len} eval_stride:{A.eval_stride} iterations:{A.iterations} warmup_steps:{A.warmup_steps} max_wallclock_seconds:{A.max_wallclock_seconds:.3f}");B(f"seed:{A.seed}");B(f"model_compile:{not A.disable_model_compile}");B(f"skip_prequant_eval_zero_iters:{A.skip_prequant_eval_zero_iters}")
	if A.init_model_path:B(f"init_model_path:{A.init_model_path}")
	if INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS:B(f"int8_keep_float_large_name_patterns:{INT8_KEEP_FLOAT_LARGE_NAME_PATTERNS}")
	B(f"qat_int4:{A.qat_int4} qat_modules:{w}");B(f"qat_only_target_params:{A.qat_only_target_params}");B(f"int4_group_size:{INT4_GROUP_SIZE if INT4_GROUP_SIZE>0 else'row'} int4_clip_percentile:{INT4_CLIP_PERCENTILE}");B(f"model_compressor:{A.model_compressor} model_compress_preset:{A.model_compress_preset} quant_pickle_protocol:{A.quant_pickle_protocol} quant_use_new_zipfile_serialization:{A.quant_use_new_zipfile_serialization} quant_load_weights_only:{A.quant_load_weights_only}");e=DistributedTokenLoader(A.train_files,P,F,G)
	def R():
		for A in H:A.zero_grad(set_to_none=_B)
	S=1e3*A.max_wallclock_seconds if A.max_wallclock_seconds>0 else _A
	def AO(step,elapsed_ms):
		C=elapsed_ms;B=step
		if A.warmdown_iters<=0:return _D
		if S is _A:F=max(A.iterations-A.warmdown_iters,0);return max((A.iterations-B)/max(A.warmdown_iters,1),_F)if F<=B<A.iterations else _D
		G=C/max(B,1);D=A.warmdown_iters*G;E=max(S-C,_F);return E/max(D,1e-09)if E<=D else _D
	if A.warmup_steps>0:
		AP={A:B.detach().cpu().clone()for(A,B)in C.state_dict().items()};AQ=[copy.deepcopy(A.state_dict())for A in H];J.train()
		for f in range(A.warmup_steps):
			R()
			for g in range(E):
				if I:J.require_backward_grad_sync=g==E-1
				h,i=e.next_batch(A.train_batch_tokens,A.train_seq_len,E)
				with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):AR=J(h,i)
				(AR*r).backward()
			for L in H:L.step()
			R()
			if A.warmup_steps<=20 or(f+1)%10==0 or f+1==A.warmup_steps:B(f"warmup_step:{f+1}/{A.warmup_steps}")
		C.load_state_dict(AP,strict=_B)
		for(L,AS)in zip(H,AQ,strict=_B):L.load_state_dict(AS)
		R()
		if I:J.require_backward_grad_sync=_B
		e=DistributedTokenLoader(A.train_files,P,F,G)
	N=_F;O=_A;torch.cuda.synchronize();W=time.perf_counter();D=0
	while _B:
		A0=D==A.iterations or O is not _A and D>=O;A1=A0 or A.val_loss_every>0 and D%A.val_loss_every==0
		if A.skip_prequant_eval_zero_iters and A.iterations==0 and D==0:A1=_C
		if A1:torch.cuda.synchronize();N+=1e3*(time.perf_counter()-W);AT,AU=eval_val(A,J,P,F,G,E,Z,t,u,v);B(f"step:{D}/{A.iterations} val_loss:{AT:.4f} val_bpb:{AU:.4f} train_time:{N:.0f}ms step_avg:{N/max(D,1):.2f}ms");torch.cuda.synchronize();W=time.perf_counter()
		if A0:
			if O is not _A and D<A.iterations:B(f"stopping_early: wallclock_cap train_time:{N:.0f}ms step:{D}/{A.iterations}")
			break
		AV=N+1e3*(time.perf_counter()-W);AW=AO(D,AV);R();j=torch.zeros((),device=G)
		for g in range(E):
			if I:J.require_backward_grad_sync=g==E-1
			h,i=e.next_batch(A.train_batch_tokens,A.train_seq_len,E)
			with torch.autocast(device_type=_I,dtype=torch.bfloat16,enabled=_B):A2=J(h,i)
			j+=A2.detach();(A2*r).backward()
		j/=E
		if K is not _A:
			A3=min(D/A.muon_momentum_warmup_steps,_D)if A.muon_momentum_warmup_steps>0 else _D;AX=(1-A3)*A.muon_momentum_warmup_start+A3*A.muon_momentum
			for M in K.param_groups:M[_P]=AX
		for L in H:
			for M in L.param_groups:M[_H]=M[T]*AW
		if A.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(C.parameters(),A.grad_clip_norm)
		for L in H:L.step()
		if K is not _A and A.muon_weight_decay>0:
			AY=_D-A.muon_weight_decay*K.param_groups[0][_H]
			for a in b:a.mul_(AY)
		R();D+=1;k=N+1e3*(time.perf_counter()-W);AZ=A.train_log_every>0 and(D<=10 or D%A.train_log_every==0 or O is not _A)
		if AZ:B(f"step:{D}/{A.iterations} train_loss:{j.item():.4f} train_time:{k:.0f}ms step_avg:{k/D:.2f}ms")
		l=S is not _A and k>=S
		if I and S is not _A:A4=torch.tensor(int(l),device=G);dist.all_reduce(A4,op=dist.ReduceOp.MAX);l=bool(A4.item())
		if O is _A and l:O=D
	B(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB")
	if U:torch.save(C.state_dict(),AC);A5=os.path.getsize(AC);X=Path(__file__).stat().st_size;B(f"Serialized model: {A5} bytes");B(f"Code size: {X} bytes");B(f"Total submission size: {A5+X} bytes")
	A6,m=quantize_state_dict_int8(C.state_dict());A7=io.BytesIO();n={}
	if A.quant_pickle_protocol>0:n[_Q]=A.quant_pickle_protocol
	if not A.quant_use_new_zipfile_serialization:n[_R]=_C
	torch.save(A6,A7,**n);Aa=A7.getvalue();Ab=save_quantized_state(A6,A);Ac=len(Aa)
	if U:
		with open(p,'wb')as o:o.write(Ab)
		A8=os.path.getsize(p);X=Path(__file__).stat().st_size;Ad=m[_M]/max(m[_G],1);B(f"Serialized model int8+{A.model_compressor}: {A8} bytes (payload:{m[_G]} raw_torch:{Ac} payload_ratio:{Ad:.2f}x)");B(f"Total submission size int8+{A.model_compressor}: {A8+X} bytes")
	if I:dist.barrier()
	with open(p,'rb')as o:Ae=o.read()
	Af=load_quantized_state(Ae,A);C.load_state_dict(dequantize_state_dict_int8(Af,C.state_dict()),strict=_B);torch.cuda.synchronize();Ag=time.perf_counter();A9,AA=eval_val(A,J,P,F,G,E,Z,t,u,v);torch.cuda.synchronize();B(f"final_int8_zlib_roundtrip val_loss:{A9:.4f} val_bpb:{AA:.4f} eval_time:{1e3*(time.perf_counter()-Ag):.0f}ms");B(f"final_int8_zlib_roundtrip_exact val_loss:{A9:.8f} val_bpb:{AA:.8f}")
	if I:dist.destroy_process_group()
if __name__=='__main__':main()