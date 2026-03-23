  用法：
  # 提取
  python3 plot/extract.py --type capgs   <log> --name capgs_bicycle
  python3 plot/extract.py --type gsscale <log> --name gsscale_bicycle

  # 画图（单个或多个对比）
  python3 plot/plot_mem.py debug/capgs_bicycle_metrics.csv
  python3 plot/plot_mem.py debug/capgs_bicycle_metrics.csv debug/gsscale_bicycle_metrics.csv