# this file is largely based on https://github.com/jakevdp/mpld3/blob/master/mpld3/_server.py
# Copyright (c) 2013, Jake Vanderplas
"""
A Simple server used to serve LDAvis visualizations
"""
import sys
import threading
import webbrowser
import socket
import itertools
import random

IPYTHON_WARNING = """
Note: if you're in the IPython notebook, pyLDAvis.show() is not the best command
      to use. Consider using pyLDAvis.display(), or pyLDAvis.enable_notebook().
      See more information at http://pyLDAvis.github.io/quickstart.html .

You must interrupt the kernel to end this command
"""

try:
    # Python 2.x
    import BaseHTTPServer as server
except ImportError:
    # Python 3.x
    from http import server


def generate_handler(html, files=None):
    if files is None:
        files = {}

    class MyHandler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            """Respond to a GET request."""
            if self.path == '/':
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write("<html><head>"
                                 "<title>LDAvis</title>"
                                 "</head><body>\n".encode())
                self.wfile.write(html.encode())
                self.wfile.write("</body></html>".encode())
            elif self.path in files:
                content_type, content = files[self.path]
                self.send_response(200)
                self.send_header("Content-type", content_type)
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self.send_error(404)

    return MyHandler


def find_open_port(ip, port, n=50):
    """Find an open port near the specified port"""
    ports = itertools.chain((port + i for i in range(n)),
                            (port + random.randint(-2 * n, 2 * n)))

    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, port))
        s.close()
        if result != 0:
            return port
    raise ValueError("no open ports found")


def serve(html, ip='127.0.0.1', port=8888, n_retries=50, files=None,
          ipython_warning=True, open_browser=True, http_server=None):
    """Start a server serving the given HTML, and (optionally) open a
    browser

    Parameters
    ----------
    html : string
        HTML to serve
    ip : string (default = '127.0.0.1')
        ip address at which the HTML will be served.
    port : int (default = 8888)
        the port at which to serve the HTML
    n_retries : int (default = 50)
        the number of nearby ports to search if the specified port is in use.
    files : dictionary (optional)
        dictionary of extra content to serve
    ipython_warning : bool (optional)
        if True (default), then print a warning if this is used within IPython
    open_browser : bool (optional)
        if True (default), then open a web browser to the given HTML
    http_server : class (optional)
        optionally specify an HTTPServer class to use for showing the
        figure. The default is Python's basic HTTPServer.
    """
    port = find_open_port(ip, port, n_retries)
    Handler = generate_handler(html, files)

    if http_server is None:
        srvr = server.HTTPServer((ip, port), Handler)
    else:
        srvr = http_server((ip, port), Handler)

    if ipython_warning:
        try:
            __IPYTHON__
        except:
            pass
        else:
            print(IPYTHON_WARNING)

    # Start the server
    print("Serving to http://{0}:{1}/    [Ctrl-C to exit]".format(ip, port))
    sys.stdout.flush()

    if open_browser:
        # Use a thread to open a web browser pointing to the server
        b = lambda: webbrowser.open('http://{0}:{1}'.format(ip, port))
        threading.Thread(target=b).start()

    try:
        srvr.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        print("\nstopping Server...")

    srvr.server_close()
