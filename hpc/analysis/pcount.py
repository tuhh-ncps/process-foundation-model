import torch
from collections import defaultdict
BB="outputs/backbones/backbone-20260825-125337-multi-none-frozen-gin15allw1-b1041f"
re=torch.load(BB+"/role_encoder.pt",map_location="cpu")
print("=== role_encoder.pt: learned vs per-log DATA buffer ===")
for k,v in re.items():
    if not hasattr(v,"numel"): continue
    shape=tuple(v.shape); n=v.numel()
    kind="DATA(per-log)" if (683 in shape) else "learned"
    print("  %-14s %-16s %9s  %s"%(k,str(shape),format(n,","),kind))
learned=sum(v.numel() for k,v in re.items() if hasattr(v,"numel") and 683 not in tuple(v.shape))
print("  --> LEARNED role params:", format(learned,","))
print()
print("=== backbone.pt breakdown ===")
bb=torch.load(BB+"/backbone.pt",map_location="cpu")
g=defaultdict(int)
for k,v in bb.items():
    if hasattr(v,"numel"): g[".".join(k.split(".")[:2])]+=v.numel()
for k,n in sorted(g.items(),key=lambda x:-x[1])[:8]: print("  %-28s %s"%(k,format(n,",")))
print("  TOTAL backbone:", format(sum(v.numel() for v in bb.values() if hasattr(v,'numel')),","))
print()
d,h=256,128
lin=lambda i,o:i*o+o
mlp=lambda i,hh,o:lin(i,hh)+lin(hh,o)
print("=== downstream task heads (d=256, head_hidden=128) ===")
for V in (20,40):
    print("  vocab V=%d: classif/set Linear(d->V)=%s | regression MLP(d->128->1)=%s"
          %(V, format(lin(d,V),","), format(mlp(d,h,1),",")))
