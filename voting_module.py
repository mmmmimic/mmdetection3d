import torch
import cython


batch_size = 15
point_num = 6
sample_num = 10
N = 100
C = 5
new_features = torch.cuda.FloatTensor(batch_size, C, point_num, sample_num)+1
grouped_xyz = torch.rand(batch_size,3,point_num,sample_num)
center_xyz = torch.rand(batch_size, point_num, 3)
idx = torch.randint(0,1,(batch_size,point_num,sample_num))
points_xyz = torch.rand(batch_size,N,3)
edge_weight = torch.zeros(batch_size,point_num,sample_num)
LAMBDA = 0.5 
grouped_xyz_copy = grouped_xyz.clone()
center_xyz_copy = center_xyz.clone()
idx_copy = idx.clone()
batch_size, _, point_num, sample_num = grouped_xyz_copy.size()
edge_weight = torch.zeros(batch_size,point_num,sample_num)
LAMBDA = 0.5

for b in range(batch_size):
    for p in range(point_num):
        # for each batch and each region
        center = center_xyz_copy[b,p,:].view(3,1).contiguous()
        # remove duplicates
        uniq_idx = torch.unique(idx_copy[b,p,:])
        uniq_num = uniq_idx.size(0)
        if uniq_num<=1:
            edge_weight[b,p,:] = torch.ones(1,1,sample_num)
            continue
        # handle the first point
        point_sel = grouped_xyz_copy[b,:,p,0].view(3,1).contiguous()
        point_clst = grouped_xyz_copy[b,:,p,1:uniq_num]
        offs = torch.norm(point_sel-center,dim=0)
        dist = torch.norm(point_clst-torch.cat(point_clst.size(1)*[point_sel],dim=1),dim=0)
        if dist.size(0)>1:
            dist = torch.min(dist) 
        edge_weight[b,p,0] = offs-LAMBDA*dist
        edge_weight[b,p,uniq_num:sample_num+1] = offs-LAMBDA*dist
        for pt in range(1,uniq_num):
            point_clst = (torch.cat([grouped_xyz_copy[b,:,p,:pt],grouped_xyz_copy[b,:,p,pt+1:]],dim=1))
            point_sel = grouped_xyz_copy[b,:,p,pt].view(3,1).contiguous()
            offs = torch.norm(point_sel-center,dim=0)
            dist = torch.norm(point_clst-torch.cat(point_clst.size(1)*[point_sel],dim=1),dim=0)
            if dist.size(0)>1:
                dist = torch.min(dist) 
            edge_weight[b,p,pt] = offs-LAMBDA*dist
        #normalization
        edge_weight[b,p,:]-= torch.min(edge_weight[b,p,:])
        edge_weight[b,p,:]/= (torch.max(edge_weight[b,p,:])-torch.min(edge_weight[b,p,:]))
        edge_weight[b,p,:]+=0.1
# output size (B,npoint,sample_num)
edge_weights = torch.cat(new_features.size(1)*[edge_weight.unsqueeze(1)],dim=1).contiguous()
edge_weights.requires_grad = True
# (B,C,npoints,nsample)
new_features = new_features.mul(edge_weights.cuda()).contiguous() 