  关键观察

  1. train 是"反例"：户外场景但 mean vis% 62.3%（全场最高），peak 2.42 GB 也高——证明内存和室内/户外无关，就是 vis% 的函数。
  2. counter / kitchen 都在 max vis% ≈ 90-98%，几乎所有点都在 frustum 里 → 所有 block 都得激活 → peak 爆。
  3. max vis% 比 mean vis% 更能预测 peak：room 和 bonsai mean 差不多（32 / 33%），但 bonsai max 高（85 vs 74）→ bonsai peak
   高 0.27 GB。
  4. 低 vis% 的 outdoor 360°（bicycle / stump） peak 都在 1.6 GB 附近，frustum 每次只抓一角。