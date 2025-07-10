import os
from chatbot.app import create_app

app = create_app()

if __name__ == '__main__':
    print('Running on : http://127.0.0.1:5000/')
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
