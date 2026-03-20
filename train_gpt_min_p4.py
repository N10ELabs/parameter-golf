from __future__ import annotations
AB=RuntimeError
h=getattr
g=FileNotFoundError
f=sorted
t=open
s=print
r=isinstance
R=tuple
q=sum
c=min
b=len
a=bool
U=any
Q=range
L=max
H=ValueError
B=float
D=int
import copy,glob as S,io,math as K,os as C,random,sys,time as T,uuid,zlib
from pathlib import Path as Z
import numpy as J,sentencepiece as AX,torch as A,torch.distributed as G,torch.nn.functional as F
from torch import Tensor,nn as E
from torch.nn.parallel import DistributedDataParallel as AY
class AZ:data_path=C.environ.get('DATA_PATH','./data/datasets/fineweb10B_sp1024');train_files=C.path.join(data_path,'fineweb_train_*.bin');val_files=C.path.join(data_path,'fineweb_val_*.bin');tokenizer_path=C.environ.get('TOKENIZER_PATH','./data/tokenizers/fineweb_1024_bpe.model');run_id=C.environ.get('RUN_ID',str(uuid.uuid4()));seed=D(C.environ.get('SEED',1337));val_batch_size=D(C.environ.get('VAL_BATCH_SIZE',524288));val_loss_every=D(C.environ.get('VAL_LOSS_EVERY',1000));train_log_every=D(C.environ.get('TRAIN_LOG_EVERY',200));iterations=D(C.environ.get('ITERATIONS',20000));warmdown_iters=D(C.environ.get('WARMDOWN_ITERS',1200));warmup_steps=D(C.environ.get('WARMUP_STEPS',20));train_batch_tokens=D(C.environ.get('TRAIN_BATCH_TOKENS',524288));train_seq_len=D(C.environ.get('TRAIN_SEQ_LEN',1024));max_wallclock_seconds=B(C.environ.get('MAX_WALLCLOCK_SECONDS',6e2));qk_gain_init=B(C.environ.get('QK_GAIN_INIT',1.5));init_model_path=C.environ.get('INIT_MODEL_PATH','');disable_model_compile=a(D(C.environ.get('DISABLE_MODEL_COMPILE','0')));qat_int4=a(D(C.environ.get('QAT_INT4','0')));qat_only_target_params=a(D(C.environ.get('QAT_ONLY_TARGET_PARAMS','0')));vocab_size=D(C.environ.get('VOCAB_SIZE',1024));num_layers=D(C.environ.get('NUM_LAYERS',9));num_kv_heads=D(C.environ.get('NUM_KV_HEADS',4));model_dim=D(C.environ.get('MODEL_DIM',512));num_heads=D(C.environ.get('NUM_HEADS',8));mlp_mult=D(C.environ.get('MLP_MULT',2));tie_embeddings=a(D(C.environ.get('TIE_EMBEDDINGS','1')));rope_base=B(C.environ.get('ROPE_BASE',1e4));logit_softcap=B(C.environ.get('LOGIT_SOFTCAP',3e1));embed_lr=B(C.environ.get('EMBED_LR',.6));head_lr=B(C.environ.get('HEAD_LR',.008));tied_embed_lr=B(C.environ.get('TIED_EMBED_LR',.05));tied_embed_init_std=B(C.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=B(C.environ.get('MATRIX_LR',.04));scalar_lr=B(C.environ.get('SCALAR_LR',.04));muon_momentum=B(C.environ.get('MUON_MOMENTUM',.95));muon_backend_steps=D(C.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=B(C.environ.get('MUON_MOMENTUM_WARMUP_START',.85));muon_momentum_warmup_steps=D(C.environ.get('MUON_MOMENTUM_WARMUP_STEPS',500));beta1=B(C.environ.get('BETA1',.9));beta2=B(C.environ.get('BETA2',.95));adam_eps=B(C.environ.get('ADAM_EPS',1e-08));grad_clip_norm=B(C.environ.get('GRAD_CLIP_NORM',.0))
def i(G,steps=10,eps=1e-07):
	D,E,F=3.4445,-4.775,2.0315;A=G.bfloat16();A/=A.norm()+eps;C=G.size(0)>G.size(1)
	if C:A=A.T
	for I in Q(steps):B=A@A.T;H=E*B+F*B@B;A=D*A+H@A
	return A.T if C else A
class Aa(A.optim.Optimizer):
	def __init__(A,params,lr,momentum,backend_steps,nesterov=True):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov))
	@A.no_grad()
	def step(self,closure=None):
		M=closure;N=None
		if M is not None:
			with A.enable_grad():N=M()
		I=G.is_available()and G.is_initialized();Q=G.get_world_size()if I else 1;R=G.get_rank()if I else 0
		for F in self.param_groups:
			H=F['params']
			if not H:continue
			S=F['lr'];O=F['momentum'];T=F['backend_steps'];U=F['nesterov'];V=q(D(A.numel())for A in H);J=A.zeros(V,device=H[0].device,dtype=A.bfloat16);E=0
			for(W,C)in enumerate(H):
				if W%Q==R and C.grad is not None:
					B=C.grad;K=self.state[C]
					if'momentum_buffer'not in K:K['momentum_buffer']=A.zeros_like(B)
					P=K['momentum_buffer'];P.mul_(O).add_(B)
					if U:B=B.add(P,alpha=O)
					B=i(B,steps=T);B*=L(1,B.size(0)/B.size(1))**.5;J[E:E+C.numel()]=B.reshape(-1)
				E+=C.numel()
			if I:G.all_reduce(J,op=G.ReduceOp.SUM)
			E=0
			for C in H:B=J[E:E+C.numel()].view_as(C).to(dtype=C.dtype);C.add_(B,alpha=-S);E+=C.numel()
		return N
def Ab(sp,vocab_size,device):
	F=device;C=sp;I=D(C.vocab_size());G=L(I,vocab_size);H=J.zeros((G,),dtype=J.int16);K=J.zeros((G,),dtype=J.bool_);M=J.ones((G,),dtype=J.bool_)
	for B in Q(I):
		if C.is_control(B)or C.is_unknown(B)or C.is_unused(B):continue
		M[B]=False
		if C.is_byte(B):H[B]=1;continue
		E=C.id_to_piece(B)
		if E.startswith('▁'):K[B]=True;E=E[1:]
		H[B]=b(E.encode('utf-8'))
	return A.tensor(H,dtype=A.int16,device=F),A.tensor(K,dtype=A.bool,device=F),A.tensor(M,dtype=A.bool,device=F)
def Ac(pattern,seq_len):
	C=pattern;B=seq_len;D=[Z(A)for A in f(S.glob(C))]
	if not D:raise g(f"No files found for pattern: {C}")
	E=A.cat([N(A)for A in D]).contiguous();F=(E.numel()-1)//B*B
	if F<=0:raise H(f"Validation split is too short for TRAIN_SEQ_LEN={B}")
	return E[:F+1]
def A8(args,model,rank,world_size,device,grad_accum_steps,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut):
	O=val_tokens;N=grad_accum_steps;I=model;E=device;D=world_size;C=args;P=C.val_batch_size//(D*N)
	if P<C.train_seq_len:raise H(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={C.val_batch_size}, WORLD_SIZE={D}, GRAD_ACCUM_STEPS={N}, TRAIN_SEQ_LEN={C.train_seq_len}")
	R=P//C.train_seq_len;S=(O.numel()-1)//C.train_seq_len;b=S*rank//D;T=S*(rank+1)//D;J=A.zeros((),device=E,dtype=A.float64);F=A.zeros((),device=E,dtype=A.float64);L=A.zeros((),device=E,dtype=A.float64);I.eval()
	with A.inference_mode():
		for U in Q(b,T,R):
			d=c(U+R,T);e=U*C.train_seq_len;f=d*C.train_seq_len+1;V=O[e:f].to(device=E,dtype=A.int64,non_blocking=True);W=V[:-1].reshape(-1,C.train_seq_len);M=V[1:].reshape(-1,C.train_seq_len)
			with A.autocast(device_type='cuda',dtype=A.bfloat16,enabled=True):g=I(W,M).detach()
			X=B(M.numel());J+=g.to(A.float64)*X;F+=X;h=W.reshape(-1);Y=M.reshape(-1);Z=base_bytes_lut[Y].to(dtype=A.int16);Z+=(has_leading_space_lut[Y]&~is_boundary_token_lut[h]).to(dtype=A.int16);L+=Z.to(A.float64).sum()
	if G.is_available()and G.is_initialized():G.all_reduce(J,op=G.ReduceOp.SUM);G.all_reduce(F,op=G.ReduceOp.SUM);G.all_reduce(L,op=G.ReduceOp.SUM)
	a=J/F;i=a.item()/K.log(2.);j=F.item()/L.item();I.train();return B(a.item()),B(i*j)
j=R(A for A in C.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights').split(',')if A)
l=R(A for A in C.environ.get('INT8_KEEP_FLOAT_FP32_NAME_PATTERNS',','.join(j)).split(',')if A)
m=65536
V=A.float16
M=A.float16
W=99.99984
X=W/1e2
A9=B(C.environ.get('INT4_CLIP_PERCENTILE',str(W)))
Y=A9/1e2
k=D(C.environ.get('INT4_GROUP_SIZE','0'))
p=R(A for A in C.environ.get('INT4_NAME_PATTERNS','mlp.fc.weight,mlp.proj.weight').split(',')if A)
def I(t):return D(t.numel())*D(t.element_size())
def n(name,t):
	if U(A in name for A in l):return t.float().contiguous()
	return t if t.dtype==V else t.to(dtype=V).contiguous()
def o(name,t):return t.ndim==2 and U(A in name for A in p)
def u(q):
	B=q.reshape(-1).to(A.int16)
	if B.numel()&1:B=A.cat((B,A.zeros(1,dtype=A.int16)))
	B=B&15;return(B[0::2]|B[1::2]<<4).to(A.uint8).contiguous()
def v(packed,numel):C=packed.reshape(-1).to(A.int16);B=A.stack((C&15,C>>4),dim=1).reshape(-1)[:numel];B=A.where(B>=8,B-16,B);return B.to(A.float32)
def d(t):
	if t.ndim!=2:raise H('int4 grouping only supports 2D tensors')
	A=t.shape[1];B=A if k<=0 else c(k,A);C=K.ceil(A/B);D=C*B-A
	if D:t=F.pad(t,(0,D))
	return t.view(t.shape[0],C,B),A
def w(scale,ref):
	B=ref;A=scale
	if A.ndim==1:return A.float().view(B.shape[0],1)
	C=K.ceil(B.shape[1]/A.shape[1]);return A.float().repeat_interleave(C,dim=1)[:,:B.shape[1]]
def y(t):
	if t.ndim!=2:return t
	B=t.float();C,F=d(B);D=A.quantile(C.abs(),Y,dim=2)if C.numel()else A.empty(C.shape[:2],dtype=A.float32,device=B.device);G=A.maximum(A.minimum(C,D[...,None]),-D[...,None]);E=(D/7.).clamp_min(1./7.);H=(A.clamp(A.round(G/E[...,None]),-7,7)*E[...,None]).reshape(B.shape[0],-1)[:,:F];return(B+(H-B).detach()).to(dtype=t.dtype)
def x(t):
	C=t.float()
	if C.ndim==2:D=A.quantile(C.abs(),X,dim=1)if C.numel()else A.empty((C.shape[0],),dtype=A.float32);G=A.maximum(A.minimum(C,D[:,None]),-D[:,None]);E=(D/127.).clamp_min(1./127.);F=A.clamp(A.round(G/E[:,None]),-127,127).to(A.int8).contiguous();return F,E.to(dtype=M).contiguous()
	D=B(A.quantile(C.abs().flatten(),X).item())if C.numel()else .0;E=A.tensor(D/127. if D>0 else 1.,dtype=A.float32);F=A.clamp(A.round(A.clamp(C,-D,D)/E),-127,127).to(A.int8).contiguous();return F,E.to(dtype=M)
def z(t):
	C=t.float()
	if C.ndim!=2:raise H('int4 path only supports 2D tensors')
	B,F=d(C);D=A.quantile(B.abs(),Y,dim=2)if B.numel()else A.empty(B.shape[:2],dtype=A.float32);G=A.maximum(A.minimum(B,D[...,None]),-D[...,None]);E=(D/7.).clamp_min(1./7.);I=A.clamp(A.round(G/E[...,None]),-7,7).to(A.int8).reshape(C.shape[0],-1)[:,:F].contiguous();return u(I),E.to(dtype=M).contiguous()
def Ad(state_dict):
	H={};J={};K={};L={};G={};A=dict.fromkeys(('param_count','num_tensors','num_float_tensors','num_nonfloat_tensors','baseline_tensor_bytes','int8_payload_bytes'),0)
	for(C,N)in state_dict.items():
		B=N.detach().to('cpu').contiguous();A['param_count']+=D(B.numel());A['num_tensors']+=1;A['baseline_tensor_bytes']+=I(B)
		if not B.is_floating_point():A['num_nonfloat_tensors']+=1;G[C]=B;A['int8_payload_bytes']+=I(B);continue
		if B.numel()<=m:M=n(C,B);G[C]=M;A['int8_payload_bytes']+=I(M);continue
		A['num_float_tensors']+=1
		if o(C,B):E,F=z(B);K[C]=E;L[C]=F;A['int8_payload_bytes']+=I(E)+I(F)
		else:E,F=x(B);H[C]=E;J[C]=F;A['int8_payload_bytes']+=I(E)+I(F)
	return{'q':H,'s':J,'q4':K,'s4':L,'p':G},A
def Ae(obj,template_state):
	H=template_state;E=obj;F={}
	for(A,D)in E['q'].items():
		C=H[A];G=E['s'][A]
		if G.ndim:F[A]=(D.float()*G.float().view(D.shape[0],*[1]*(D.ndim-1))).to(dtype=C.dtype).contiguous()
		else:F[A]=(D.float()*B(G.item())).to(dtype=C.dtype).contiguous()
	for(A,D)in E.get('q4',{}).items():C=H[A];G=E['s4'][A];I=v(D,C.numel()).view_as(C);F[A]=(I*w(G,C)).to(dtype=C.dtype).contiguous()
	for(A,J)in E['p'].items():C=H[A];F[A]=J.detach().to('cpu').to(dtype=C.dtype).contiguous()
	return F
def N(file):
	B=file;F=256*J.dtype('<i4').itemsize;K=J.dtype('<u2').itemsize;C=J.fromfile(B,dtype='<i4',count=256)
	if C.size!=256 or D(C[0])!=20240520 or D(C[1])!=1:raise H(f"Unexpected shard header for {B}")
	E=D(C[2]);G=F+E*K
	if B.stat().st_size!=G:raise H(f"Shard size mismatch for {B}: expected {G} bytes")
	I=J.fromfile(B,dtype='<u2',count=E,offset=F)
	if I.size!=E:raise H(f"Short read for {B}")
	return A.from_numpy(I.astype(J.uint16,copy=False))
class A0:
	def __init__(A,pattern):
		B=pattern;A.files=[Z(A)for A in f(S.glob(B))]
		if not A.files:raise g(f"No files found for pattern: {B}")
		A.file_idx=0;A.tokens=N(A.files[0]);A.pos=0
	def _advance_file(A):A.file_idx=(A.file_idx+1)%b(A.files);A.tokens=N(A.files[A.file_idx]);A.pos=0
	def take(B,n):
		C=[];D=n
		while D>0:
			F=B.tokens.numel()-B.pos
			if F<=0:B._advance_file();continue
			E=c(D,F);C.append(B.tokens[B.pos:B.pos+E]);B.pos+=E;D-=E
		return C[0]if b(C)==1 else A.cat(C)
class AA:
	def __init__(A,pattern,rank,world_size,device):A.rank=rank;A.world_size=world_size;A.device=device;A.stream=A0(pattern)
	def next_batch(B,global_tokens,seq_len,grad_accum_steps):D=seq_len;G=global_tokens//(B.world_size*grad_accum_steps);C=G+1;H=B.stream.take(C*B.world_size);E=B.rank*C;F=H[E:E+C].to(dtype=A.int64);I=F[:-1].reshape(-1,D);J=F[1:].reshape(-1,D);return I.to(B.device,non_blocking=True),J.to(B.device,non_blocking=True)
class O(E.Module):
	def __init__(A,eps=None):super().__init__();A.eps=eps
	def forward(A,x):return F.rms_norm(x,(x.size(-1),),eps=A.eps)
class P(E.Linear):
	def forward(A,x):B=A.bias.to(x.dtype)if A.bias is not None else None;C=y(A.weight)if A.training and h(A,'_qat_int4',False)else A.weight;return F.linear(x,C.to(x.dtype),B)
def Af(module):
	with A.no_grad():
		for(C,B)in module.named_parameters():
			if(B.ndim<2 or U(A in C for A in j))and B.dtype!=A.float32:B.data=B.data.float()
class A1(E.Module):
	def __init__(B,dim,base=1e4):super().__init__();C=1./base**(A.arange(0,dim,2,dtype=A.float32)/dim);B.register_buffer('inv_freq',C,persistent=False);B._seq_len_cached=0;B._cos_cached=None;B._sin_cached=None
	def forward(B,seq_len,device,dtype):
		E=dtype;D=device;C=seq_len
		if B._cos_cached is None or B._sin_cached is None or B._seq_len_cached!=C or B._cos_cached.device!=D:G=A.arange(C,device=D,dtype=B.inv_freq.dtype);F=A.outer(G,B.inv_freq.to(D));B._cos_cached=F.cos()[None,None,:,:];B._sin_cached=F.sin()[None,None,:,:];B._seq_len_cached=C
		return B._cos_cached.to(dtype=E),B._sin_cached.to(dtype=E)
def e(x,cos,sin):B=x.size(-1)//2;C,D=x[...,:B],x[...,B:];return A.cat((C*cos+D*sin,C*-sin+D*cos),dim=-1)
class A2(E.Module):
	def __init__(B,dim,num_heads,num_kv_heads,rope_base,qk_gain_init):
		F=num_kv_heads;D=num_heads;C=dim;super().__init__()
		if C%D!=0:raise H('model_dim must be divisible by num_heads')
		if D%F!=0:raise H('num_heads must be divisible by num_kv_heads')
		B.num_heads=D;B.num_kv_heads=F;B.head_dim=C//D
		if B.head_dim%2!=0:raise H('head_dim must be even for RoPE')
		G=B.num_kv_heads*B.head_dim;B.c_q=P(C,C,bias=False);B.c_k=P(C,G,bias=False);B.c_v=P(C,G,bias=False);B.proj=P(C,C,bias=False);B.proj._zero_init=True;B.q_gain=E.Parameter(A.full((D,),qk_gain_init,dtype=A.float32));B.rotary=A1(B.head_dim,base=rope_base)
	def forward(A,x):E,D,J=x.shape;B=A.c_q(x).reshape(E,D,A.num_heads,A.head_dim).transpose(1,2);C=A.c_k(x).reshape(E,D,A.num_kv_heads,A.head_dim).transpose(1,2);K=A.c_v(x).reshape(E,D,A.num_kv_heads,A.head_dim).transpose(1,2);B=F.rms_norm(B,(B.size(-1),));C=F.rms_norm(C,(C.size(-1),));H,I=A.rotary(D,x.device,B.dtype);B=e(B,H,I);C=e(C,H,I);B=B*A.q_gain.to(dtype=B.dtype)[None,:,None,None];G=F.scaled_dot_product_attention(B,C,K,attn_mask=None,is_causal=True,enable_gqa=A.num_kv_heads!=A.num_heads);G=G.transpose(1,2).contiguous().reshape(E,D,J);return A.proj(G)
class A3(E.Module):
	def __init__(A,dim,mlp_mult):B=dim;super().__init__();C=mlp_mult*B;A.fc=P(B,C,bias=False);A.proj=P(C,B,bias=False);A.proj._zero_init=True
	def forward(B,x):x=A.relu(B.fc(x));return B.proj(x.square())
class A4(E.Module):
	def __init__(B,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init):C=dim;super().__init__();B.attn_norm=O();B.mlp_norm=O();B.attn=A2(C,num_heads,num_kv_heads,rope_base,qk_gain_init);B.mlp=A3(C,mlp_mult);B.attn_scale=E.Parameter(A.ones(C,dtype=A.float32));B.mlp_scale=E.Parameter(A.ones(C,dtype=A.float32));B.resid_mix=E.Parameter(A.stack((A.ones(C),A.zeros(C))).float())
	def forward(A,x,x0):B=A.resid_mix.to(dtype=x.dtype);x=B[0][None,None,:]*x+B[1][None,None,:]*x0;C=A.attn(A.attn_norm(x));x=x+A.attn_scale.to(dtype=x.dtype)[None,None,:]*C;x=x+A.mlp_scale.to(dtype=x.dtype)[None,None,:]*A.mlp(A.mlp_norm(x));return x
class Ag(E.Module):
	def __init__(B,vocab_size,num_layers,model_dim,num_heads,num_kv_heads,mlp_mult,tie_embeddings,tied_embed_init_std,logit_softcap,rope_base,qk_gain_init):
		I=tie_embeddings;G=vocab_size;F=logit_softcap;D=num_layers;C=model_dim;super().__init__()
		if F<=.0:raise H(f"logit_softcap must be positive, got {F}")
		B.tie_embeddings=I;B.tied_embed_init_std=tied_embed_init_std;B.logit_softcap=F;B.tok_emb=E.Embedding(G,C);B.num_encoder_layers=D//2;B.num_decoder_layers=D-B.num_encoder_layers;B.num_skip_weights=c(B.num_encoder_layers,B.num_decoder_layers);B.skip_weights=E.Parameter(A.ones(B.num_skip_weights,C,dtype=A.float32));B.blocks=E.ModuleList([A4(C,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init)for A in Q(D)]);B.final_norm=O();B.lm_head=None if I else P(C,G,bias=False)
		if B.lm_head is not None:B.lm_head._zero_init=True
		B._init_weights()
	def _init_weights(A):
		if A.tie_embeddings:E.init.normal_(A.tok_emb.weight,mean=.0,std=A.tied_embed_init_std)
		for B in A.modules():
			if r(B,E.Linear)and h(B,'_zero_init',False):E.init.zeros_(B.weight)
	def forward(C,input_ids,target_ids):
		B=C.tok_emb(input_ids);B=F.rms_norm(B,(B.size(-1),));G=B;E=[]
		for D in Q(C.num_encoder_layers):B=C.blocks[D](B,G);E.append(B)
		for D in Q(C.num_decoder_layers):
			if E:B=B+C.skip_weights[D].to(dtype=B.dtype)[None,None,:]*E.pop()
			B=C.blocks[C.num_encoder_layers+D](B,G)
		B=C.final_norm(B).reshape(-1,B.size(-1));I=target_ids.reshape(-1)
		if C.tie_embeddings:H=F.linear(B,C.tok_emb.weight)
		else:
			if C.lm_head is None:raise AB('lm_head is required when tie_embeddings=False')
			H=C.lm_head(B)
		J=C.logit_softcap*A.tanh(H/C.logit_softcap);return F.cross_entropy(J.float(),I,reduction='mean')
def A5():
	global i;B=AZ();i=A.compile(i);R='RANK'in C.environ and'WORLD_SIZE'in C.environ;d=D(C.environ.get('RANK','0'));M=D(C.environ.get('WORLD_SIZE','1'));AC=D(C.environ.get('LOCAL_RANK','0'))
	if M<=0:raise H(f"WORLD_SIZE must be positive, got {M}")
	if 8%M!=0:raise H(f"WORLD_SIZE={M} must divide 8 so grad_accum_steps stays integral")
	K=8//M;AD=1./K
	if not A.cuda.is_available():raise AB('CUDA is required')
	N=A.device('cuda',AC);A.cuda.set_device(N)
	if R:G.init_process_group(backend='nccl',device_id=N);G.barrier()
	l=d==0;A.backends.cuda.matmul.allow_tf32=True;A.backends.cudnn.allow_tf32=True;from torch.backends.cuda import enable_cudnn_sdp as Ah,enable_flash_sdp as Ai,enable_math_sdp as Aj,enable_mem_efficient_sdp as Ak;Ah(False);Ai(True);Ak(False);Aj(False);m=None
	if l:C.makedirs('logs',exist_ok=True);m=f"logs/{B.run_id}.txt";s(m)
	def E(msg,console=True):
		if not l:return
		if console:s(msg)
		if m is not None:
			with t(m,'a',encoding='utf-8')as A:s(msg,file=A)
	random.seed(B.seed);J.random.seed(B.seed);A.manual_seed(B.seed);A.cuda.manual_seed_all(B.seed)
	if not B.tokenizer_path.endswith('.model'):raise H(f"Script only setup for SentencePiece .model file: {B.tokenizer_path}")
	u=AX.SentencePieceProcessor(model_file=B.tokenizer_path)
	if D(u.vocab_size())!=B.vocab_size:raise H(f"VOCAB_SIZE={B.vocab_size} does not match tokenizer vocab_size={D(u.vocab_size())}")
	AE=Z(B.data_path).resolve();Al=b(list(AE.glob('fineweb_train_*.bin')));v=Ac(B.val_files,B.train_seq_len);AF,AG,AH=Ab(u,B.vocab_size,N);E(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={B.tokenizer_path}");E(f"train_loader:dataset:{AE.name} train_shards:{Al}");E(f"val_loader:shards pattern={B.val_files} tokens:{v.numel()-1}");F=Ag(vocab_size=B.vocab_size,num_layers=B.num_layers,model_dim=B.model_dim,num_heads=B.num_heads,num_kv_heads=B.num_kv_heads,mlp_mult=B.mlp_mult,tie_embeddings=B.tie_embeddings,tied_embed_init_std=B.tied_embed_init_std,logit_softcap=B.logit_softcap,rope_base=B.rope_base,qk_gain_init=B.qk_gain_init).to(N).bfloat16()
	for e in F.modules():
		if r(e,P):e.float()
	Af(F)
	if B.init_model_path:Am=A.load(B.init_model_path,map_location='cpu');F.load_state_dict(Am,strict=True)
	AI=0
	for(An,e)in F.named_modules():
		if r(e,P):AJ=B.qat_int4 and U(A in f"{An}.weight"for A in p);e._qat_int4=AJ;AI+=D(AJ)
	if B.qat_only_target_params:
		for(Ao,Ap)in F.named_parameters():Ap.requires_grad_(U(A in Ao for A in p))
	AK=F if B.disable_model_compile else A.compile(F,dynamic=False,fullgraph=True);S=AY(AK,device_ids=[AC],broadcast_buffers=False)if R else AK;AL=list(F.blocks.named_parameters());AM=[A for(B,A)in AL if A.requires_grad and A.ndim==2 and not U(A in B for A in j)];w=[A for(B,A)in AL if A.requires_grad and(A.ndim<2 or U(A in B for A in j))]
	if F.skip_weights.requires_grad and F.skip_weights.numel()>0:w.append(F.skip_weights)
	x=B.tied_embed_lr if B.tie_embeddings else B.embed_lr;O=[]
	if F.tok_emb.weight.requires_grad:O.append(A.optim.Adam([{'params':[F.tok_emb.weight],'lr':x,'base_lr':x}],betas=(B.beta1,B.beta2),eps=B.adam_eps,fused=True))
	if F.lm_head is not None and F.lm_head.weight.requires_grad:O.append(A.optim.Adam([{'params':[F.lm_head.weight],'lr':B.head_lr,'base_lr':B.head_lr}],betas=(B.beta1,B.beta2),eps=B.adam_eps,fused=True))
	f=None
	if AM:
		f=Aa(AM,lr=B.matrix_lr,momentum=B.muon_momentum,backend_steps=B.muon_backend_steps)
		for W in f.param_groups:W['base_lr']=B.matrix_lr
		O.append(f)
	if w:O.append(A.optim.Adam([{'params':w,'lr':B.scalar_lr,'base_lr':B.scalar_lr}],betas=(B.beta1,B.beta2),eps=B.adam_eps,fused=True))
	Aq=q(A.numel()for A in F.parameters());Ar=q(A.numel()for A in F.parameters()if A.requires_grad);E(f"model_params:{Aq}");E(f"trainable_params:{Ar}");E(f"world_size:{M} grad_accum_steps:{K}");E('sdp_backends:cudnn=False flash=True mem_efficient=False math=False');E(f"attention_mode:gqa num_heads:{B.num_heads} num_kv_heads:{B.num_kv_heads}");E(f"tie_embeddings:{B.tie_embeddings} embed_lr:{x} head_lr:{B.head_lr if F.lm_head is not None else .0} matrix_lr:{B.matrix_lr} scalar_lr:{B.scalar_lr}");E(f"train_batch_tokens:{B.train_batch_tokens} train_seq_len:{B.train_seq_len} iterations:{B.iterations} warmup_steps:{B.warmup_steps} max_wallclock_seconds:{B.max_wallclock_seconds:.3f}");E(f"seed:{B.seed}");E(f"model_compile:{not B.disable_model_compile}")
	if B.init_model_path:E(f"init_model_path:{B.init_model_path}")
	E(f"qat_int4:{B.qat_int4} qat_modules:{AI}");E(f"qat_only_target_params:{B.qat_only_target_params}");E(f"int4_group_size:{k if k>0 else 'row'} int4_clip_percentile:{A9}");y=AA(B.train_files,d,M,N)
	def g():
		for A in O:A.zero_grad(set_to_none=True)
	h=1e3*B.max_wallclock_seconds if B.max_wallclock_seconds>0 else None
	def As(step,elapsed_ms):
		C=elapsed_ms;A=step
		if B.warmdown_iters<=0:return 1.
		if h is None:F=L(B.iterations-B.warmdown_iters,0);return L((B.iterations-A)/L(B.warmdown_iters,1),.0)if F<=A<B.iterations else 1.
		G=C/L(A,1);D=B.warmdown_iters*G;E=L(h-C,.0);return E/L(D,1e-09)if E<=D else 1.
	if B.warmup_steps>0:
		At={A:B.detach().cpu().clone()for(A,B)in F.state_dict().items()};Au=[copy.deepcopy(A.state_dict())for A in O];S.train()
		for z in Q(B.warmup_steps):
			g()
			for A0 in Q(K):
				if R:S.require_backward_grad_sync=A0==K-1
				A1,A2=y.next_batch(B.train_batch_tokens,B.train_seq_len,K)
				with A.autocast(device_type='cuda',dtype=A.bfloat16,enabled=True):Av=S(A1,A2)
				(Av*AD).backward()
			for V in O:V.step()
			g()
			if B.warmup_steps<=20 or(z+1)%10==0 or z+1==B.warmup_steps:E(f"warmup_step:{z+1}/{B.warmup_steps}")
		F.load_state_dict(At,strict=True)
		for(V,Aw)in zip(O,Au,strict=True):V.load_state_dict(Aw)
		g()
		if R:S.require_backward_grad_sync=True
		y=AA(B.train_files,d,M,N)
	X=.0;Y=None;A.cuda.synchronize();n=T.perf_counter();I=0
	while True:
		AN=I==B.iterations or Y is not None and I>=Y;Ax=AN or B.val_loss_every>0 and I%B.val_loss_every==0
		if Ax:A.cuda.synchronize();X+=1e3*(T.perf_counter()-n);Ay,Az=A8(B,S,d,M,N,K,v,AF,AG,AH);E(f"step:{I}/{B.iterations} val_loss:{Ay:.4f} val_bpb:{Az:.4f} train_time:{X:.0f}ms step_avg:{X/L(I,1):.2f}ms");A.cuda.synchronize();n=T.perf_counter()
		if AN:
			if Y is not None and I<B.iterations:E(f"stopping_early: wallclock_cap train_time:{X:.0f}ms step:{I}/{B.iterations}")
			break
		A_=X+1e3*(T.perf_counter()-n);B0=As(I,A_);g();A3=A.zeros((),device=N)
		for A0 in Q(K):
			if R:S.require_backward_grad_sync=A0==K-1
			A1,A2=y.next_batch(B.train_batch_tokens,B.train_seq_len,K)
			with A.autocast(device_type='cuda',dtype=A.bfloat16,enabled=True):AO=S(A1,A2)
			A3+=AO.detach();(AO*AD).backward()
		A3/=K
		if f is not None:
			AP=c(I/B.muon_momentum_warmup_steps,1.)if B.muon_momentum_warmup_steps>0 else 1.;B1=(1-AP)*B.muon_momentum_warmup_start+AP*B.muon_momentum
			for W in f.param_groups:W['momentum']=B1
		for V in O:
			for W in V.param_groups:W['lr']=W['base_lr']*B0
		if B.grad_clip_norm>0:A.nn.utils.clip_grad_norm_(F.parameters(),B.grad_clip_norm)
		for V in O:V.step()
		g();I+=1;A4=X+1e3*(T.perf_counter()-n);B2=B.train_log_every>0 and(I<=10 or I%B.train_log_every==0 or Y is not None)
		if B2:E(f"step:{I}/{B.iterations} train_loss:{A3.item():.4f} train_time:{A4:.0f}ms step_avg:{A4/I:.2f}ms")
		A5=h is not None and A4>=h
		if R and h is not None:AQ=A.tensor(D(A5),device=N);G.all_reduce(AQ,op=G.ReduceOp.MAX);A5=a(AQ.item())
		if Y is None and A5:Y=I
	E(f"peak memory allocated: {A.cuda.max_memory_allocated()//1024//1024} MiB reserved: {A.cuda.max_memory_reserved()//1024//1024} MiB")
	if l:A.save(F.state_dict(),'final_model.pt');AR=C.path.getsize('final_model.pt');o=Z(__file__).stat().st_size;E(f"Serialized model: {AR} bytes");E(f"Code size: {o} bytes");E(f"Total submission size: {AR+o} bytes")
	B3,A6=Ad(F.state_dict());AS=io.BytesIO();A.save(B3,AS,pickle_protocol=4);AT=AS.getvalue();B4=zlib.compress(AT,level=9);B5=b(AT)
	if l:
		with t('final_model.int8.ptz','wb')as A7:A7.write(B4)
		AU=C.path.getsize('final_model.int8.ptz');o=Z(__file__).stat().st_size;B6=A6['baseline_tensor_bytes']/L(A6['int8_payload_bytes'],1);E(f"Serialized model int8+zlib: {AU} bytes (payload:{A6['int8_payload_bytes']} raw_torch:{B5} payload_ratio:{B6:.2f}x)");E(f"Total submission size int8+zlib: {AU+o} bytes")
	if R:G.barrier()
	with t('final_model.int8.ptz','rb')as A7:B7=A7.read()
	B8=A.load(io.BytesIO(zlib.decompress(B7)),map_location='cpu');F.load_state_dict(Ae(B8,F.state_dict()),strict=True);A.cuda.synchronize();B9=T.perf_counter();AV,AW=A8(B,S,d,M,N,K,v,AF,AG,AH);A.cuda.synchronize();E(f"final_int8_zlib_roundtrip val_loss:{AV:.4f} val_bpb:{AW:.4f} eval_time:{1e3*(T.perf_counter()-B9):.0f}ms");E(f"final_int8_zlib_roundtrip_exact val_loss:{AV:.8f} val_bpb:{AW:.8f}")
	if R:G.destroy_process_group()
if __name__=='__main__':A5()
