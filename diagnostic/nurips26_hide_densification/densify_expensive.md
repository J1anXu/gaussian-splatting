                                                 
问题定位                                                                            
                                                                                    
densify_and_prune 每 100 iter 触发一次（iter 600, 700, 800...直到 15000），每次 6 个
block 串行执行，每个 ~400-480ms，总计 2.55s 完全阻塞训练。                         
                                                                                    
相比之下，正常一个 iter 才 ~120ms，所以每次 densify 等于浪费了 ~21 个 iter 的时间。 
                                                                                    
为什么这么慢？                                                                      
                                                            
每个 block 的 densify_and_prune 做了这些事：                                        
                                                            
1. _sync_packed_adam_to_optimizer_state() — 6 个参数组 .clone() 大tensor            
2. densify_and_clone() — cat 操作（扩展 optimizer state）
3. densify_and_split() — cat + bmm + prune                                          
4. prune_points() — boolean 索引，重建所有 optimizer state                          
5. torch.cuda.empty_cache() — 强制 GPU 同步 + 缓存清理，这个可能单独就 100-200ms    
6. 然后 pack_to_buffer() — 重建 packed buffer + _init_packed_adam_state() 重新分配  
pinned memory                                                                       
                                                                                    
对于 ~800K 点/block，D=59 的 packed buffer，每个 .clone() / torch.cat() 都是在 CPU  
上操作 ~180MB 的 tensor。                                  
                                                                                    
能否后台化？你的直觉是对的，可以分三级做：                                          

Level 1: 立即收益（不改语义，0脏数据）                                              
                                                            
a) 去掉 per-block 的 torch.cuda.empty_cache()                                       
                                                            
gaussian_model.py:735 每个 block 的 densify_and_prune 最后都调了一次。6个 block =   
6次 empty_cache，每次都强制 GPU sync。改为只在所有 block   
做完后调一次，或者干脆不调（CUDA allocator 自己会复用）。                           
                                                            
b) 多线程并行 densify 所有 block                                                    

6 个 block 的 densify_and_prune 完全独立（各自的 GaussianModel 实例），可以用       
ThreadPoolExecutor 并行。因为是 CPU-bound 且大量用了 PyTorch C++ backend（释放
GIL），并行度不错。从 2.55s → ~500ms。                                              
                                                            
Level 2: 延迟执行（1-2 iter 脏数据，训练影响极小）                                  

核心思路：densify 不在当前 iter 同步执行，而是 启动后台线程，主线程继续训练。       
                                                            
iter 700:  标记 "需要 densify"，snapshot stats → 启动后台线程                       
iter 701:  用旧点云正常训练 ← 脏数据，但只差 1 iter 的 densify 结果                 
iter 702:  后台完成 → swap in 新模型                                                
                                                                                    
为什么脏数据影响小？                                                                
- densify_stats 本来就是累积 100 个 iter 的梯度统计，少 1-2 iter 的统计几乎无影响
- clone/split/prune 改变的点数通常只占总点数的 1-3%                                 
- 训练用的是 _packed buffer，只要在 swap 时做 atomic 替换即可
                                                                                    
需要处理的约束：                                                                    
- 后台 densify 期间，该 block 的 densify_stats 要暂停（accumulators 被 reset 了）   
- 后台 densify 期间，该 block 的 adam step 要用旧的 idx（点数没变）                 
- swap 时需要重建 packed buffer + pinned buffer + frustum cache 失效
                                                                                    
Level 3: 完全异步（激进）                                                           
                                                                                    
densify 每 100 iter 触发 → 改为 每 block 独立调度，scatter 到不同 iter。比如 6 个   
block，每个间隔 ~17 iter 分开 densify，每次只 densify 1 个 
block，完全隐藏在流水线里。                                                         
                                                            
---
我建议先做 Level 1，因为收益确定且风险为零：
                                                                                    
1. empty_cache() 只调一次 → 预计省 500-1000ms
2. 多线程并行 → 从 2.55s 降到 ~500ms                                                
3. 两个加起来可能从 2.55s → 200-400ms                                               
                                                                                    
