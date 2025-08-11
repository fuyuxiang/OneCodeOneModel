from flask import Flask, request, jsonify, render_template
import logging
from pathlib import Path
from .generator import PoetryGenerator
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PoetryAPI")

# 创建Flask应用
app = Flask(__name__)

# 初始化生成器
try:
    # 默认使用最佳模型
    generator = PoetryGenerator(Path("saved_models/best_poetry_model.pth"))
    logger.info("诗歌生成器初始化成功")
except Exception as e:
    logger.error(f"初始化生成器失败: {e}")
    generator = None


@app.route('/')
def index():
    """展示Web界面"""
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate_poetry():
    """生成诗歌API"""
    if generator is None:
        return jsonify({"error": "生成器未初始化"}), 500

    try:
        # 获取请求参数
        data = request.json
        start_str = data.get('start_str', '')
        max_length = int(data.get('max_length', 100))
        temperature = float(data.get('temperature', 1.0))

        # 生成诗歌
        raw_poem = generator.generate(start_str, max_length, temperature)
        formatted_poem = generator.format_poem(raw_poem)

        # 返回结果
        return jsonify({
            "status": "success",
            "raw_poem": raw_poem,
            "formatted_poem": formatted_poem,
            "start_str": start_str
        })

    except Exception as e:
        logger.error(f"生成诗歌时出错: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/demo', methods=['GET'])
def demo_generate():
    """演示接口，随机生成诗歌"""
    if generator is None:
        return jsonify({"error": "生成器未初始化"}), 500

    try:
        # 随机选择起始词（常见诗歌开头）
        starters = ["春", "秋", "山", "水", "月", "花", "风", "云", "夜", "日"]
        start_str = np.random.choice(starters)

        # 生成诗歌
        raw_poem = generator.generate(start_str, 60, 0.8)
        formatted_poem = generator.format_poem(raw_poem)

        return jsonify({
            "status": "success",
            "formatted_poem": formatted_poem,
            "start_str": start_str
        })

    except Exception as e:
        logger.error(f"演示生成时出错: {e}")
        return jsonify({"error": str(e)}), 500
