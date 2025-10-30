"""
打字数据收集器 - API版本
使用公开API获取随机文本，更稳定
(修改点)
- 移除诗词API
- 新增爬取长句功能：source=crawl
"""

from flask import Flask, render_template_string, request, jsonify
import json
import requests
import random
import re

# 尝试引入 bs4；若没有安装也能工作（用正则清洗HTML）
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

app = Flask(__name__)
collected_data = []

class APITextSource:
    """文本来源（API + 本地 + 抓取长句）"""

    # 一些默认可抓取的文章页面（中文/英文混合，尽量通用公开页）
    CRAWL_SEEDS = [
        # 你可以换成自己更稳定的来源
        "https://www.wenzizhan.com/article/10015.html",
        
    ]

    def get_hitokoto(self):
        """一言API - 随机句子（较短）"""
        try:
            url = 'https://v1.hitokoto.cn/'
            params = {'c': random.choice(['a', 'b', 'd', 'i', 'k'])}
            resp = requests.get(url, params=params, timeout=3)
            data = resp.json()
            return data.get('hitokoto', '')
        except:
            return None

    def get_yiyan(self):
        """易言API - 随机励志句子（较短）"""
        try:
            url = 'https://api.oick.cn/yulu/api.php'
            resp = requests.get(url, timeout=3)
            return resp.text.strip()
        except:
            return None

    def get_local_mixed(self):
        """本地混合文本 - 多为较长片段，作为兜底"""
        texts = [
            # 技术类长文本（略）
            "使用Python开发AI应用，结合深度学习框架实现智能化系统",
            "深度学习DeepLearning框架PyTorch能够快速构建神经网络模型",
            "自然语言处理NLP技术在智能客服和机器翻译中应用广泛",
            "机器学习MachineLearning算法包括监督学习和无监督学习两大类",
            "数据科学DataScience分析需要掌握统计学和编程技能",
            "PyTorch深度学习训练支持GPU加速，可以显著提升模型训练速度",
            "Transformer架构解析：自注意力机制是其核心创新点",
            "大语言模型LLM应用开发包括Prompt工程和Fine-tuning微调技术",
            "React前端开发实战项目中使用Hooks和Context进行状态管理",
            "Docker容器化部署可以保证应用在不同环境中的一致性运行",
            "Kubernetes集群管理K8s支持自动扩缩容和服务发现功能",
            "Redis缓存优化技巧包括合理设置过期时间和使用管道批量操作",
            "微服务架构设计要考虑服务拆分粒度和服务间通信方式的选择",
            "RESTful API设计遵循资源导向原则，使用HTTP动词表示操作",
            "敏捷开发Agile方法论强调快速迭代和持续交付价值给客户",
            # 日常生活/中英混合（略）
            "今天天气真不错，阳光明媚适合出门散步放松心情",
            "坚持就是胜利，每天进步一点点终将达成目标实现梦想",
            "创新驱动发展，只有不断创新才能在竞争中保持领先优势",
            "用户体验第一，产品设计要从用户需求出发注重细节打磨",
            "在GitHub上参与OpenSource开源项目可以提升编程能力和协作经验",
            "Machine学习模型训练完成后需要进行Validation验证和Testing测试",
            "Cloud云计算平台AWS和Azure提供了丰富的PaaS和IaaS服务",
            "Continuous Integration持续集成和Continuous Deployment持续部署提升交付效率",
        ]
        return random.choice(texts)

    # ---------- 抓取长句：核心实现 ----------
    def _fetch_html(self, url: str, timeout: float = 5.0) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
        return resp.text

    def _html_to_text(self, html: str) -> str:
        # 优先用 BeautifulSoup 更干净
        if HAS_BS4:
            soup = BeautifulSoup(html, "html.parser")
            # 去除脚本/样式
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        else:
            # 简单正则去标签（保底）
            text = re.sub(r"(?is)<script.*?>.*?</script>", "", html)
            text = re.sub(r"(?is)<style.*?>.*?</style>", "", text)
            text = re.sub(r"(?is)<[^>]+>", "", text)

        # 压缩多余空白
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text).strip()
        return text

    def _split_sentences(self, text: str):
        """
        粗粒度切句：支持中英混排。
        - 中文：以 。！？； 断句
        - 英文：以 .!? 分句（考虑保留原标点）
        """
        # 先统一换行 → 空格，避免打断句号附近的分割
        text = text.replace("\n", " ").strip()

        # 中文切句（保留标点）
        cn_parts = re.split(r"([。！？；])", text)
        merged = []
        for i in range(0, len(cn_parts), 2):
            seg = cn_parts[i].strip()
            punct = cn_parts[i + 1] if i + 1 < len(cn_parts) else ""
            if seg:
                merged.append(seg + punct)

        # 对每个中文段再做英文句子切分（保留标点）
        sentences = []
        for seg in merged:
            parts = re.split(r"([.!?])", seg)
            buf = ""
            for j in range(0, len(parts), 2):
                s = parts[j].strip()
                p = parts[j + 1] if j + 1 < len(parts) else ""
                if s:
                    buf = (s + p).strip()
                    sentences.append(buf)
        # 去掉过短垃圾片段
        sentences = [s.strip() for s in sentences if len(s.strip()) >= 3]
        return sentences

    def _pick_long_sentence(self, sentences, min_len=20, max_len=120):
        """优先挑选 20~120 字/字符的句子，若无则退而求其次挑最长的一条"""
        candidates = [s for s in sentences if min_len <= len(s) <= max_len]
        if candidates:
            return random.choice(candidates)
        # 退而求其次：从所有句子中挑长度靠前的前若干
        sentences = sorted(sentences, key=lambda s: len(s), reverse=True)
        return sentences[0] if sentences else None

    def get_long_sentence_crawl(self, url: str = None):
        """爬取网页正文，抽取长句"""
        try:
            seed = url or random.choice(self.CRAWL_SEEDS)
            html = self._fetch_html(seed, timeout=6.0)
            text = self._html_to_text(html)
            sentences = self._split_sentences(text)
            long_one = self._pick_long_sentence(sentences)
            return long_one
        except Exception:
            return None

    # ---------- 统一入口 ----------
    def get_text(self, source='mixed', url: str = None):
        """获取文本 - 带降级策略"""
        if source == 'hitokoto':
            text = self.get_hitokoto()
            if text and 5 <= len(text) <= 60:
                return text
        elif source == 'yiyan':
            text = self.get_yiyan()
            if text and 5 <= len(text) <= 120:
                return text
        elif source == 'crawl':
            text = self.get_long_sentence_crawl(url=url)
            if text and len(text) >= 20:
                return text

        # 默认 / 失败：本地长文本
        return self.get_local_mixed()

api_source = APITextSource()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>打字数据收集器</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }
        h1 { color: #333; margin-bottom: 20px; text-align: center; }
        
        .source-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 12px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 10px 20px;
            border: 2px solid #667eea;
            background: white;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .tab:hover { background: #f0f0f0; }
        .tab.active { background: #667eea; color: white; }

        .crawl-row {
            display: flex; gap: 8px; margin-bottom: 10px; align-items: center;
        }
        .crawl-row input {
            flex: 1; padding: 10px; border: 2px solid #667eea; border-radius: 8px;
        }
        .crawl-row button {
            padding: 10px 16px; border-radius: 8px; border: none; background:#667eea; color:#fff; cursor:pointer;
        }
        .crawl-row small { color:#555; }
        
        #target {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin: 20px 0;
            font-size: 24px;
            border-radius: 10px;
            text-align: center;
            min-height: 70px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        #input-area {
            width: 100%;
            height: 150px;
            font-size: 20px;
            padding: 15px;
            border: 3px solid #667eea;
            border-radius: 8px;
            resize: none;
            font-family: inherit;
        }
        #input-area:focus {
            outline: none;
            border-color: #764ba2;
            box-shadow: 0 0 15px rgba(102,126,234,0.3);
        }
        
        .buttons {
            margin-top: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary { background: #667eea; color: white; }
        .btn-success { background: #4CAF50; color: white; }
        .btn-danger { background: #ff6b6b; color: white; }
        .btn:hover { transform: translateY(-2px); }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .stat-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 28px; font-weight: bold; color: #667eea; }
        .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
        
        #log {
            max-height: 300px;
            overflow-y: auto;
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            margin-top: 20px;
            border-radius: 8px;
            font-family: 'Consolas', monospace;
            font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 打字数据收集器</h1>
        
        <div class="source-tabs">
            <div class="tab active" onclick="changeSource('mixed')">本地文本</div>
            <div class="tab" onclick="changeSource('hitokoto')">一言句子</div>
            <div class="tab" onclick="changeSource('yiyan')">励志语录</div>
            <div class="tab" onclick="changeSource('crawl')">长句抓取</div>
        </div>

        <div class="crawl-row" id="crawl-row" style="display:none;">
            <input id="crawl-url" placeholder="可选：指定一个文章URL来抓取长句（留空则随机种子）">
            <button onclick="fetchNewTarget()">从URL抓取</button>
            <small>抓取失败会自动回退到随机种子或本地文本</small>
        </div>
        
        <div id="target">正在加载...</div>
        
        <textarea id="input-area" placeholder="开始输入..."></textarea>
        
        <div class="buttons">
            <button class="btn btn-success" onclick="saveCurrent()">✓ 保存当前</button>
            <button class="btn btn-success" onclick="saveAndNext()">✓ 保存并下一个</button>
            <button class="btn btn-primary" onclick="refreshTarget()">🔄 刷新文本</button>
            <button class="btn btn-danger" onclick="downloadData()">💾 下载数据</button>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-value" id="count">0</div>
                <div class="stat-label">已保存</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="steps">0</div>
                <div class="stat-label">总步数</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="ime-steps">0</div>
                <div class="stat-label">拼音步数</div>
            </div>
        </div>
        
        <div id="log"></div>
    </div>

    <script>
        let currentTarget = '';
        let currentSource = 'mixed';
        let typingLog = [];
        let startTime = null;
        let lastContent = '';
        let isComposing = false;
        let imeStepCount = 0;
        
        const inputArea = document.getElementById('input-area');
        const logDiv = document.getElementById('log');
        const crawlRow = document.getElementById('crawl-row');
        const crawlUrl = document.getElementById('crawl-url');
        
        function addLog(msg) {
            logDiv.innerHTML += msg + '<br>';
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function updateStats() {
            document.getElementById('steps').textContent = typingLog.length;
            document.getElementById('ime-steps').textContent = imeStepCount;
        }
        
        function changeSource(source) {
            currentSource = source;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            // 切换抓取源时显示URL栏
            crawlRow.style.display = (source === 'crawl') ? 'flex' : 'none';
            fetchNewTarget();
        }
        
        function fetchNewTarget() {
            let url = '/get_text?source=' + currentSource;
            if (currentSource === 'crawl') {
                const u = crawlUrl.value.trim();
                if (u) url += '&url=' + encodeURIComponent(u);
            }
            fetch(url)
                .then(r => r.json())
                .then(data => {
                    currentTarget = data.text;
                    resetInput();
                });
        }
        
        function resetInput() {
            document.getElementById('target').textContent = currentTarget;
            inputArea.value = '';
            typingLog = [];
            startTime = null;
            lastContent = '';
            isComposing = false;
            imeStepCount = 0;
            logDiv.innerHTML = '等待输入...<br>';
            updateStats();
            inputArea.focus();
        }
        
        // 输入法事件
        inputArea.addEventListener('compositionstart', () => {
            isComposing = true;
            if (!startTime) startTime = Date.now();
        });
        
        inputArea.addEventListener('compositionupdate', (e) => {
            if (!startTime) startTime = Date.now();
            const time = (Date.now() - startTime) / 1000;
            typingLog.push({
                time: time,
                action: 'ime_update',
                content: e.data
            });
            addLog(`[${time.toFixed(2)}s] 拼音: ${e.data}`);
            imeStepCount++;
            updateStats();
        });
        
        inputArea.addEventListener('compositionend', (e) => {
            isComposing = false;
            if (!startTime) startTime = Date.now();
            const time = (Date.now() - startTime) / 1000;
            const current = inputArea.value;
            if (current.length > lastContent.length) {
                const newText = current.slice(lastContent.length);
                typingLog.push({
                    time: time,
                    action: 'type',
                    content: newText,
                    method: 'ime'
                });
                addLog(`[${time.toFixed(2)}s] 输入: ${newText}`);
                updateStats();
            }
            lastContent = current;
        });
        
        inputArea.addEventListener('input', () => {
            if (isComposing) return;
            if (!startTime) startTime = Date.now();
            const time = (Date.now() - startTime) / 1000;
            const current = inputArea.value;
            
            if (current.length > lastContent.length) {
                const newText = current.slice(lastContent.length);
                typingLog.push({
                    time: time,
                    action: 'type',
                    content: newText,
                    method: 'direct'
                });
                addLog(`[${time.toFixed(2)}s] 直接: ${newText}`);
                updateStats();
            } else if (current.length < lastContent.length) {
                const deleted = lastContent.slice(current.length);
                typingLog.push({
                    time: time,
                    action: 'delete',
                    deleted: deleted
                });
                addLog(`[${time.toFixed(2)}s] 删除: ${deleted}`);
                updateStats();
            }
            lastContent = current;
        });
        
        function saveCurrent() {
            if (typingLog.length === 0) {
                alert('请先输入内容！');
                return;
            }
            
            fetch('/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    target: currentTarget,
                    final_output: inputArea.value,
                    typing_log: typingLog,
                    source: currentSource
                })
            }).then(r => r.json()).then(result => {
                document.getElementById('count').textContent = result.total;
                alert(`✅ 已保存！\\n总数: ${result.total}\\n\\n提示：可以点击"刷新文本"继续输入新的内容`);
            }).catch(err => {
                alert('保存失败: ' + err);
            });
        }
        
        function saveAndNext() {
            if (typingLog.length === 0) {
                alert('请先输入内容！');
                return;
            }
            
            fetch('/save', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    target: currentTarget,
                    final_output: inputArea.value,
                    typing_log: typingLog,
                    source: currentSource
                })
            }).then(r => r.json()).then(result => {
                document.getElementById('count').textContent = result.total;
                // 保存成功后自动获取下一个
                fetchNewTarget();
            }).catch(err => {
                alert('保存失败: ' + err);
            });
        }
        
        function refreshTarget() {
            if (typingLog.length > 0) {
                if (!confirm('当前输入的内容还未保存，确定要刷新吗？')) {
                    return;
                }
            }
            fetchNewTarget();
        }
        
        function nextTarget() {
            refreshTarget();
        }
        
        function downloadData() {
            fetch('/download').then(r => r.json()).then(data => {
                if (data.length === 0) {
                    alert('还没有数据！');
                    return;
                }
                const blob = new Blob([JSON.stringify(data, null, 2)], 
                    {type: 'application/json'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'typing_data_with_pinyin.json';
                a.click();
                alert(`已下载 ${data.length} 条数据`);
            });
        }
        
        window.onload = () => fetchNewTarget();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_text')
def get_text():
    source = request.args.get('source', 'mixed')
    url = request.args.get('url', '').strip() or None
    text = api_source.get_text(source, url=url)
    return jsonify({'text': text, 'source': source, 'url': url})

@app.route('/save', methods=['POST'])
def save():
    data = request.json
    collected_data.append(data)
    
    with open('typing_data.json', 'w', encoding='utf-8') as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n[{len(collected_data)}] {data['target']} -> {data['final_output']}")
    print(f"    步数: {len(data['typing_log'])}, 来源: {data['source']}")
    
    return jsonify({'status': 'ok', 'total': len(collected_data)})

@app.route('/download')
def download():
    return jsonify(collected_data)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("打字数据收集器 - API版本")
    print("="*60)
    print("\n文本来源:")
    print("  • 本地文本 - 离线可用")
    print("  • 一言API - 随机句子")
    print("  • 励志语录 - 随机语句")
    print("  • 长句抓取 - 抓取网页正文提取更长的句子（可在输入框指定URL）")
    print("\n使用方法:")
    print("  1. 打开 http://localhost:5000")
    print("  2. 选择文本来源（可选 crawl 并填 URL）")
    print("  3. 开始输入")
    print("  4. 保存数据 / 保存并下一个")
    print("  5. 收集20-30条后下载")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, port=5000)
