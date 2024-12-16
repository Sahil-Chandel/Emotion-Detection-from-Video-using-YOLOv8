# Purpose : REST API for processing videos and returning results.


from flask import Flask, request, jsonify
from src.pipeline.emotion_pipeline import process_video

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    video = request.files['video']
    video_path = "./input/input_video.avi"
    output_path = "./output/result.avi"
    video.save(video_path)
    process_video(video_path, output_path)
    return jsonify({"status": "success", "output": output_path})

if __name__ == "__main__":
    app.run(debug=True)