from flask import Flask, render_template, send_from_directory, jsonify, request, redirect, url_for
from clustering import getElbow, Clustering
from simulation import simulation

app = Flask(__name__)

@app.route('/get-elbow')
def get_elbow():
    return getElbow()

@app.route('/clustering')
def api_clustering():
    Clustering()
    return jsonify({"OK"})

@app.route('/simulation')
def api_simulation():
    pop_size = int(request.args.get('pop_size'))
    total_time = int(request.args.get('total_time'))
    start_day = int(request.args.get('start_day'))
    start_month = int(request.args.get('start_month'))
    res = simulation(pop_size=pop_size, total_time=total_time, start_day=start_day, start_month=start_month)
    return jsonify(res)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
