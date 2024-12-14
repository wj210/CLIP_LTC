
import numpy as np
import torch
from collections import defaultdict


class PRSLogger(object):
    def __init__(self, model, device, spatial: bool = True,attentions_edit=None,mlps_edit=None,attn_pos=None,mlp_pos=None,causal_type = None):
        self.current_layer = 0
        self.device = device
        self.attentions = []
        self.mlps = []
        self.spatial = spatial
        self.post_ln_std = None
        self.post_ln_mean = None
        self.attn_post= []
        self.post = []
        self.model = model
        self.attentions_edit = attentions_edit,
        self.mlps_edit = mlps_edit,
        self.attn_pos = attn_pos,
        self.mlp_pos = mlp_pos,
        self.causal_type = causal_type

    @torch.no_grad()
    def compute_attentions_spatial(self, ret):
        assert len(ret.shape) == 5, "Verify that you use method=`head` and not method=`head_no_spatial`" # [b, n, m, h, d]
        assert self.spatial, "Verify that you use method=`head` and not method=`head_no_spatial`"
        bias_term = self.model.visual.transformer.resblocks[
            self.current_layer
        ].attn.out_proj.bias
        self.current_layer += 1
        return_value = ret[:, 0].detach().cpu()  # This is only for the cls token
        self.attentions.append(
            return_value
            + bias_term[np.newaxis, np.newaxis, np.newaxis].cpu()
            / (return_value.shape[1] * return_value.shape[2])
        )  # [b, n, h, d]
        return ret
    
    @torch.no_grad()
    def compute_attentions_non_spatial(self, ret):
        assert len(ret.shape) == 4, "Verify that you use method=`head_no_spatial` and not method=`head`" # [b, n, h, d]
        assert not self.spatial, "Verify that you use method=`head_no_spatial` and not method=`head`"
        bias_term = self.model.visual.transformer.resblocks[
            self.current_layer
        ].attn.out_proj.bias
        self.current_layer += 1
        return_value = ret[:, 0].detach().cpu()  # This is only for the cls token
        self.attentions.append(
            return_value
            + bias_term[np.newaxis, np.newaxis].cpu()
            / (return_value.shape[1])
        )  # [b, h, d]
        return ret

    @torch.no_grad()
    def compute_mlps(self, ret):
        self.mlps.append(ret[:, 0].detach().cpu())  # [b, d]
        return ret
    
    @torch.no_grad()
    def compute_attn_post(self,ret):
        self.attn_post.append(ret[:,0].detach().cpu())
        return ret

    @torch.no_grad()
    def log_post_ln_mean(self, ret):
        self.post_ln_mean = ret.detach().cpu()  # [b, 1]
        return ret

    @torch.no_grad()
    def log_post_ln_std(self, ret):
        self.post_ln_std = ret.detach().cpu()  # [b, 1]
        return ret

    def _normalize_mlps(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]
        # This is just the normalization layer:
        mean_centered = (
            self.mlps
            - self.post_ln_mean[:, :, np.newaxis].to(self.device) / len_intermediates
        )
        weighted_mean_centered = (
            self.model.visual.ln_post.weight.detach().to(self.device) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis
        ].to(self.device)
        bias_term = (
            self.model.visual.ln_post.bias.detach().to(self.device) / len_intermediates
        )
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.visual.proj.detach().to(self.device)

    def _normalize_attentions_spatial(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1 (b,l,n,h,d)
        normalization_term = (
            self.attentions.shape[2] * self.attentions.shape[3]
        )  # n * h
        # This is just the normalization layer:
        mean_centered = self.attentions - self.post_ln_mean[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device) / (len_intermediates * normalization_term)
        weighted_mean_centered = (
            self.model.visual.ln_post.weight.detach().to(self.device) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis, np.newaxis
        ].to(self.device)
        bias_term = self.model.visual.ln_post.bias.detach().to(self.device) / (
            len_intermediates * normalization_term
        )
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.visual.proj.detach().to(self.device)

    def _normalize_attentions_non_spatial(self):
        len_intermediates = self.attentions.shape[1] + self.mlps.shape[1]  # 2*l + 1
        normalization_term = (
            self.attentions.shape[2]
        )  # h
        # This is just the normalization layer:
        mean_centered = self.attentions - self.post_ln_mean[
            :, :, np.newaxis, np.newaxis
        ].to(self.device) / (len_intermediates * normalization_term)
        weighted_mean_centered = (
            self.model.visual.ln_post.weight.detach().to(self.device) * mean_centered
        )
        weighted_mean_by_std = weighted_mean_centered / self.post_ln_std[
            :, :, np.newaxis, np.newaxis
        ].to(self.device)
        bias_term = self.model.visual.ln_post.bias.detach().to(self.device) / (
            len_intermediates * normalization_term
        )
        post_ln = weighted_mean_by_std + bias_term
        return post_ln @ self.model.visual.proj.detach().to(self.device)

    @torch.no_grad()
    def finalize(self, representation):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        self.attentions = torch.stack(self.attentions, axis=1).to(
            self.device
        )  # [b, l, n, h, d]
        self.mlps = torch.stack(self.mlps, axis=1).to(self.device)  # [b, l + 1, d]
        if self.spatial:
            projected_attentions = self._normalize_attentions_spatial()
        else:
            projected_attentions = self._normalize_attentions_non_spatial()
        projected_mlps = self._normalize_mlps()
        norm = representation.norm(dim=-1).detach()
        if self.spatial:
            return (
                projected_attentions
                / norm[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis],
                projected_mlps / norm[:, np.newaxis, np.newaxis],
            )
        return (
            projected_attentions
            / norm[:, np.newaxis, np.newaxis, np.newaxis],
            projected_mlps / norm[:, np.newaxis, np.newaxis],
        )
    
    @torch.no_grad()
    def finalize_wo_project(self):
        """We calculate the post-ln scaling, project it and normalize by the last norm."""
        self.attentions = torch.stack(self.attentions, axis=1) 
        self.mlps = torch.stack(self.mlps, axis=1)  # [b, l + 1, d]
        return self.attentions.numpy(),self.mlps.numpy()
    
    def reinit(self):
        self.current_layer = 0
        self.attentions = []
        self.mlps = []
        self.post_ln_mean = None
        self.post_ln_std = None
        torch.cuda.empty_cache()



def hook_prs_logger(model, device, spatial: bool = True,direct=False):
    """Hooks a projected residual stream logger to the model."""
    prs = PRSLogger(model, device, spatial=spatial)
    if spatial:
        model.hook_manager.register(
            "visual.transformer.resblocks.*.attn.out.post", prs.compute_attentions_spatial
        )
    else:
        model.hook_manager.register(
            "visual.transformer.resblocks.*.attn.out.post", prs.compute_attentions_non_spatial
        )
    model.hook_manager.register(
        "visual.transformer.resblocks.*.mlp.c_proj.post", prs.compute_mlps
    )
    if direct: # if measuring direct effect, only need attn and mlps
        return prs 
    model.hook_manager.register("visual.ln_pre_post", prs.compute_mlps)
    model.hook_manager.register("visual.ln_post.mean", prs.log_post_ln_mean)
    model.hook_manager.register("visual.ln_post.sqrt_var", prs.log_post_ln_std)
    # model.hook_manager.register("visual.transformer.resblocks.*.post", prs.compute_attn_post)

    return prs


class PRSLoggerEdit(object):
    def __init__(self,device,attn_val,attn_pos,mlp_val=None,mlp_pos=None,attn_bias=None):
        self.device = device
        self.attn_pos = attn_pos # only for attn head and total effect, else for indirect, we just change the entire val.
        self.attn_val = attn_val
        self.mlp_val = mlp_val
        self.mlp_pos = mlp_pos
        self.attn_bias = attn_bias
    
    @torch.no_grad()
    def edit_total(self,ret):
        assert ret.ndim == 5, f"Expected 5D tensor, got {ret.ndim}"
        bias_norm = self.attn_bias / (ret.shape[2]*ret.shape[3]) # divide by q and h
        ret  = ret + bias_norm
        ret = ret.sum(axis= 2) # sum over keys
        ret[:,0,self.attn_pos] = self.attn_val.repeat(ret.shape[0],1,1).to(self.device) # lhs is (b,n,h,d), rhs is just (1,d), only edit cls
        # ret[:,0,self.attn_pos] = torch.zeros((ret.shape[0],ret.shape[3])).to(self.device) 
        return ret.sum(axis=2)
    
    @torch.no_grad()
    def store_clean_before_edit(self,ret):
        assert ret.ndim == 5, f"Expected 5D tensor, got {ret.ndim}"
        bias_norm = self.attn_bias / (ret.shape[2]*ret.shape[3]) # divide by q and h
        self.clean_before = (ret + bias_norm).sum(axis=2) # sum over keys
        return self.clean_before.sum(axis=2) # sum over heads
    
    @torch.no_grad()
    def edit_indirect(self,ret,residual):
        assert self.clean_before is not None, "Call store_clean_before_edit before calling edit_indirect"
        assert self.clean_before.ndim == 4, f"Expected 4D tensor, got {self.clean_before.ndim}"
        self.clean_before[:,0,self.attn_pos] = self.attn_val.repeat(ret.shape[0],1,1).to(self.device) # lhs is (b,n,h,d), rhs is just (1,d)
        # self.clean_before[:,0,self.attn_pos] = torch.zeros((self.clean_before.shape[0],self.clean_before.shape[3])).to(self.device)
        return_val = self.clean_before.sum(axis=2) # sum over the head
        assert return_val.ndim == 3, f"Expected 3D tensor, got {return_val.ndim}"
        self.clean_before = None
        return residual + return_val

def hook_prs_logger_indirect(model,device,edit_heads,causal_type,mean_activations,hook_type):
    attn_val = torch.from_numpy(mean_activations['attn']) # (1,l,h,d)
    mlp_val= torch.from_numpy(mean_activations['mlp']) # (1,l,d)

    ## grp the edit heads by layer
    grpped_layers = defaultdict(list)
    for (layer,head) in edit_heads:
            grpped_layers[layer].append(head)
    
    for (layer,head) in sorted(grpped_layers.items(),key = lambda x: x[0]):
        prs = PRSLoggerEdit(device,
                    attn_val=attn_val[:,layer,head], 
                    mlp_val=mlp_val[:,layer], 
                    attn_pos=head, # only head pos
                    mlp_pos=None,
                    attn_bias = model.visual.transformer.resblocks[layer].attn.out_proj.bias
                    )
        if causal_type == 'total':
            model.hook_manager.register_single(f"visual.transformer.resblocks.{layer}.attn.out.post",prs.edit_total)
        else:
            model.hook_manager.register_single(f"visual.transformer.resblocks.{layer}.attn.out.post",prs.store_clean_before_edit)
            model.hook_manager.register_single(f"visual.transformer.resblocks.{layer}.after_attn",prs.edit_indirect)
    




    

