"""Minimum squared displacement under nonnegativity and one linear margin constraint."""
def project(h,d,goal,iterations=60):
 import torch
 # Solve v=max(h+lambda*d,0), d.v=goal, lambda>=0; goal must exceed d.h.
 dtype=h.dtype;h=h.double();d=d.double();goal=goal.double()
 feasible=(d>0).any(1)|(goal<0)
 lo=torch.zeros_like(goal);hi=torch.ones_like(goal)
 def score(lam):return (d*(h+lam[:,None]*d).clamp_min(0)).sum(1)
 for _ in range(60):
  need=(score(hi)<goal)&feasible
  if not need.any():break
  hi=torch.where(need,hi*2,hi)
 feasible&=score(hi)>=goal
 for _ in range(iterations):
  mid=(lo+hi)/2;below=score(mid)<goal;lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
 v=(h+hi[:,None]*d).clamp_min(0);v[~feasible]=h[~feasible]
 err=((d*v).sum(1)-goal).abs();assert (err[feasible]<1e-6*(1+goal[feasible].abs())).all()
 return v.to(dtype),feasible,hi

def self_test():
 import torch
 h=torch.tensor([[1.,2.],[1.,2.],[1.,2.]])
 d=torch.tensor([[1.,0.],[1.,-1.],[-1.,-1.]])
 goal=torch.tensor([2.,3.,1.]);v,f,l=project(h,d,goal)
 assert f.tolist()==[True,True,False]
 assert torch.allclose(v[0],torch.tensor([2.,2.]))
 assert torch.allclose(v[1],torch.tensor([3.,0.]),atol=1e-6)
 assert (v>=0).all()
 print('PASS: unconstrained solution, active nonnegative constraint, infeasible target')
if __name__=='__main__':self_test()
