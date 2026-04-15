import os
import torch
import numpy as np
import pandas as pd
import timesfm
import matplotlib.pyplot as plt

def main():
    print("=== 阶段一：环境与模型加载 ===")
    # 针对现代CPU优化矩阵乘法精度
    torch.set_float32_matmul_precision("high")
    
    # 实例化2.5版本的200M模型，并自动下载/加载预训练权重
    # 请确保网络畅通，初次运行会自动下载权重文件
    model_name = "google/timesfm-2.5-200m-pytorch"
    print(f"Loading TimesFM model from: {model_name}")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)

    print("=== 阶段二：定义预测规则（配置阶段） ===")
    horizon_len = 30  # 假设我们需要预测未来30个时间点
    context_len = 512 # 我们让模型“回头看”过去512个时间点
    
    # 编译模型配置
    model.compile(
        timesfm.ForecastConfig(
            max_context=context_len,
            max_horizon=horizon_len,
            normalize_inputs=True,           # 自动归一化输入，无需人工缩放数据
            use_continuous_quantile_head=True, # 开启连续分位数头，用于输出置信区间
            force_flip_invariance=True,
            infer_is_positive=False,         # 数据是否全部为正数（根据业务实际情况调整）
            fix_quantile_crossing=True,      # 修复分位数交叉问题
        )
    )

    print("=== 阶段三：准备业务数据 ===")
    train_file = "train.parquet"
    test_file = "test.parquet"
    
    # 获取历史训练数据（取最后 context_len 个点作为上下文）
    if os.path.exists(train_file):
        print(f"Loading training data from {train_file}...")
        df_train = pd.read_parquet(train_file)
        
        # 按照数据格式：第0列为时间，奇数列(1,3,5...)为close，偶数列(2,4,6...)为volume
        close_cols = df_train.columns[1::2] # 从第1列开始，步长为2
        vol_cols = df_train.columns[2::2]   # 从第2列开始，步长为2
        
        print(f"检测到 {len(close_cols)} 支股票的 close 数据。")
        
        # 为了演示绘图，这里我们提取第一支股票（即第1列）的 close 价格作为预测目标
        # 如果你想预测所有股票，可以使用循环构建一个 list: [df_train[col].values[-context_len:] for col in close_cols]
        target_col = close_cols[0]
        print(f"选择预测目标列: {target_col}")
        
        # 提取该股票的历史序列，并截取最后 context_len 个点
        history_input = df_train[target_col].values[-context_len:]
    else:
        print(f"Warning: {train_file} 未找到。生成模拟的正弦波数据作为历史数据。")
        history_input = np.sin(np.linspace(0, 20, context_len)) + np.random.normal(0, 0.1, context_len)

    # 获取测试数据（Ground Truth，用于和预测值对比）
    if os.path.exists(test_file):
        print(f"Loading test data from {test_file}...")
        df_test = pd.read_parquet(test_file)
        
        # 同样提取对应的第一支股票的 close 价格作为真实对比值
        close_cols_test = df_test.columns[1::2]
        target_col_test = close_cols_test[0]
        
        # 截取与 horizon_len 对应长度的测试数据
        test_truth = df_test[target_col_test].values[:horizon_len]
    else:
        print(f"Warning: {test_file} 未找到。生成模拟的正弦波数据作为测试真实值。")
        test_truth = np.sin(np.linspace(20, 20 + (20/context_len)*horizon_len, horizon_len))

    print("=== 阶段四：执行推理（预测阶段） ===")
    print(f"Forecasting {horizon_len} steps into the future...")
    
    # 调用 forecast 接口，输入必须是 list of 1D arrays
    point_forecast, quantile_forecast = model.forecast(
        horizon=horizon_len,
        inputs=[history_input], # 我们这里只预测一条时间序列
    )

    # 获取第一条序列的预测结果
    pred_mean = point_forecast[0]             # 形状: (horizon_len,)
    pred_quantiles = quantile_forecast[0]     # 形状: (horizon_len, num_quantiles)
    
    print("=== 阶段五：解析与使用结果（绘图保存） ===")
    plt.figure(figsize=(12, 6))
    
    # 绘制过去100个点的历史数据，方便查看趋势衔接
    plot_context = min(100, len(history_input))
    hist_x = np.arange(-plot_context, 0)
    plt.plot(hist_x, history_input[-plot_context:], label="History (Train Data)", color="black")
    
    # 绘制未来的真实测试数据
    future_x = np.arange(0, horizon_len)
    plt.plot(future_x, test_truth, label="Ground Truth (Test Data)", color="green", linestyle="--")
    
    # 绘制模型预测的点（均值）
    plt.plot(future_x, pred_mean, label="TimesFM Point Forecast", color="blue", linewidth=2)
    
    # 绘制预测的置信区间（10% 到 90% 分位数）
    # TimesFM 2.5 默认返回10个分位数: 
    # index 0: Mean, index 1: 10th, ..., index 5: 50th(Median), ..., index 9: 90th
    if pred_quantiles.shape[1] >= 10:
        plt.fill_between(
            future_x, 
            pred_quantiles[:, 1],   # 10th percentile (10% 分位数)
            pred_quantiles[:, 9],   # 90th percentile (90% 分位数)
            color="blue", 
            alpha=0.2, 
            label="Confidence Interval (10%-90%)"
        )
    
    # 画一条红色的垂直虚线分隔历史和未来
    plt.axvline(0, color="red", linestyle=":", label="Forecast Start")
    
    plt.title("TimesFM Forecast vs Ground Truth", fontsize=14)
    plt.xlabel("Time Steps")
    plt.ylabel("Target Value")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    output_plot = "forecast_result.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"✅ 推理完成！预测结果图表已保存至: {os.path.abspath(output_plot)}")

if __name__ == "__main__":
    main()
