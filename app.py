from flask import Flask, request, jsonify
from selector  import run_selector

app = Flask(__name__)

@app.route('/select-lines', methods=['POST'])
def select_lines():
    try:
        data = request.get_json()
        required = ['lat1', 'long1', 'lat2', 'long2', 'Cost', 'time', 'Comfort']
        for f in required:
            if f not in data:
                return jsonify({"error": f"Missing field: {f}"}), 400

        result = run_selector({
            'lat1':    data['lat1'],
            'long1':   data['long1'],
            'lat2':    data['lat2'],
            'long2':   data['long2'],
            'Cost':    data['Cost'],
            'time':    data['time'],
            'Comfort': data['Comfort'],
        })
        return jsonify({"data": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)