Fix A — 最小改动：merge 完立即释放 nograd 张量                               
  # train.py:415 之后，在进入 grad 阶段之前加：                                
  merge_res = merge_opt_kid(rendered_list, depth_list, alpha_list)             
  del rendered_list, depth_list, alpha_list                                    
  del all_rendered, all_depth, all_alpha                                       
  这一下省下 ~280 MB（counter 3.73 → ~3.45 GB）。副作用：无。                  
                                                                               
  Fix B — 更根本：把 merge_opt_kid 返回的 C_sorted 改成不保留 K 份，只保留     
  final_rgb + 每个 block 的增量 δ。或者把 grad 阶段改成和 merge                
  "流水线化"，每处理完一块就把那块对应的 C_sorted/prefix_T 切片释放。          
                                                                               
  Fix C — 降低 K：只对 vis% > 50% 的场景，config.NUM_BLOCKS 降到 4，peak       
  直接砍一半。                                        