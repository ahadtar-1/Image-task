"""
The module comprises of the driver program used to run the application
"""

from apis import app

if __name__ == '__main__':
    app.run(port = 5000, debug = True)