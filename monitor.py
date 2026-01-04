import time
import requests
import argparse
import sys
import unicodedata
from prometheus_client.parser import text_string_to_metric_families

def get_visual_width(s):
    """计算字符串在终端的视觉宽度（考虑中文字符占用2格）"""
    width = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ('W', 'F'):
            width += 2
        else:
            width += 1
    return width

def align_text(s, width, side='left'):
    """根据视觉宽度进行对齐填充"""
    current_width = get_visual_width(s)
    padding = " " * max(0, width - current_width)
    if side == 'left':
        return s + padding
    return padding + s

class VLLMMonitor:
    CORE_METRICS = {
        "vllm:num_requests_running": ("正在运行请求", "当前正在 GPU 上进行推理的任务数"),
        "vllm:num_requests_waiting": ("排队等待请求", "因为资源不足而在队列中排队的任务数"),
        "vllm:num_requests_swapped": ("换出请求", "因显存压力被临时移动到 CPU 内存的任务数"),
        "vllm:gpu_cache_usage_perc": ("GPU KV缓存占用", "GPU 显存中 KV Cache 的已使用百分比"),
        "vllm:cpu_cache_usage_perc": ("CPU 缓存占用", "Swap 显存的已使用百分比"),
        "vllm:avg_prompt_throughput_tok_s": ("Prompt 吞吐量", "平均每秒处理的输入 Token 数 (TTFT 相关)"),
        "vllm:avg_generation_throughput_tok_s": ("生成 吞吐量", "平均每秒生成的输出 Token 数 (TPOT 相关)"),
        "vllm:request_success_total": ("成功任务总数", "自启动以来成功处理完成的请求总计数"),
    }

    def __init__(self, url, interval):
        self.url = url
        self.interval = interval

    def fetch(self):
        try:
            r = requests.get(self.url, timeout=2)
            return r.text if r.status_code == 200 else None
        except:
            return None

    def display(self, raw_data):
        data = {}
        if raw_data:
            for family in text_string_to_metric_families(raw_data):
                if family.name in self.CORE_METRICS:
                    data[family.name] = family.samples[0].value

        # 清屏并重置光标到顶部
        sys.stdout.write("\033[H\033[J") 
        
        header = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] vLLM 核心运行状态监控, 刷新间隔: {self.interval}s"
        print(header)
        print("=" * 75)
        
        # 打印表头，确保对齐
        col1 = align_text("指标名称", 20)
        col2 = align_text("数值", 15)
        col3 = "说明"
        print(f"{col1} | {col2} | {col3}")
        print("-" * 75)
        
        for name, (label, desc) in self.CORE_METRICS.items():
            val = data.get(name, 0)
            
            # 格式化数值
            if "perc" in name:
                formatted_val = f"{val * 100:.1f}%" if val <= 1.0 else f"{val:.1f}%"
            elif "tok_s" in name:
                formatted_val = f"{val:.2f} t/s"
            else:
                formatted_val = f"{int(val)}"
                
            c1 = align_text(label, 20)
            c2 = align_text(formatted_val, 15)
            print(f"{c1} | {c2} | {desc}")
            
        print("=" * 75)
        if not raw_data:
            print("\n[警告] 无法连接到 vLLM 指标接口，请检查服务是否已启动...")

    def run(self):
        try:
            while True:
                raw = self.fetch()
                self.display(raw)
                time.sleep(self.interval)
        except KeyboardInterrupt:
            print("\n监控已停止。")

def main():
    parser = argparse.ArgumentParser(description="vLLM Metrics Monitor with Alignment")
    parser.add_argument("--url", type=str, default="http://localhost:8000/metrics", help="Metrics URL")
    parser.add_argument("--interval", type=int, default=5, help="刷新间隔(秒)，默认5s")
    args = parser.parse_args()

    VLLMMonitor(args.url, args.interval).run()

if __name__ == "__main__":
    main()
