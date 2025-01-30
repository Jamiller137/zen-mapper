"""
Mapper Visualization Tool 🎨

A visualization toolkit for my zenmapper results,
turning complex high-dimensional data into interpretable network graphs.

Warning: May cause excessive graph staring.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from flask import Flask, Response, render_template
from jinja2 import Environment, FileSystemLoader
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

class FileChangeHandler(FileSystemEventHandler):
    """
    File watcher for hot-reloading Python files.

    Monitors Python files for changes and handles automatic reloading
    of the application when modifications are detected.

    Args:
        app: Flask application instance
        main_module_path: Path to the main module being watched
    """
    def __init__(self, app, main_module_path):
        self.last_modified = time.time()
        self.app = app
        self.main_module_path = main_module_path
        self.module_name = os.path.splitext(os.path.basename(main_module_path))[0]

    def on_modified(self, event):
        """
        Handles file modification events.

        Reloads the application code when changes are detected,
        with basic debouncing to prevent multiple reloads.
        """
        if hasattr(event, 'src_path') and event.src_path.endswith('.py'):
            current_time = time.time()
            if current_time - self.last_modified > 1:
                self.last_modified = current_time
                try:
                    # clear module cache
                    for key in list(sys.modules.keys()):
                        if key.startswith(self.module_name):
                            del sys.modules[key]

                    # reload and execute
                    with open(self.main_module_path) as f:
                        code = compile(f.read(), self.main_module_path, 'exec')
                        global_dict = {
                            '__name__': '__main__',
                            '__file__': self.main_module_path,
                        }
                        exec(code, global_dict)

                    self.app.changed = True
                    print("Script reloaded and re-executed")
                except Exception as e:
                    print(f"Error reloading script: {e}")
                    import traceback
                    traceback.print_exc()

class MapperVisualizer:
    """Interactive visualization tool for Mapper results.

    Converts high-dimensional mapper output into interactive 2D network visualizations.
    Supports both dynamic (Flask-based) and static HTML rendering.

    Parameters
    ----------
        result (MapperOutput): Mapper algorithm output containing nodes, edges, and 
            simplicial complex information.

        X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).

        lens (numpy.ndarray): Filter-function-applied data used in mapper algorithm.

        X_names (list[str]): Feature names corresponding to input data columns.

        lens_names (list[str]): Names of the lens/filter functions applied.

        color_function (str | callable): Strategy for coloring nodes. Can be:
            - 'density': Color by point density
            - 'mean': Color by mean of filter values
            - callable: Custom function for node coloring

        colorscale (list | None): Custom color scale for nodes. Defaults to None.

    Returns
    -------
        MapperVisualization: Visualization object that can be rendered as HTML.

    Examples
    --------
        >>> mapper_result = run_mapper(data)
        >>> viz = MapperVisualization(
        ...     result=mapper_result,
        ...     X=data,
        ...     lens=filter_values,
        ...     X_names=['feature1', 'feature2'],
        ...     lens_names=['filter1'],
        ...     color_function='density'
        ... )
        >>> viz.to_html('output.html')
    """
    _instances = []

    def __init__(self, result, X: np.ndarray, lens: np.ndarray,
                 X_names: list = None, lens_names: list = None,
                 color_function: str = "mean", colorscale: list = None) -> None:
        self.result = result
        self.X = X
        self.lens = lens
        self.X_names = X_names or []
        self.lens_names = lens_names or []
        self.color_function = color_function
        self.colorscale = colorscale or [
            [0.0, "#440154"], [1.0, "#FDE724"]  # viridis?? please!
        ]
        self._data = None
        MapperVisualizer._instances.append(self)

    @classmethod
    def update_current_instance(cls, result, X=None, lens=None):
        """Updates the most recent visualizer instance with new data"""
        if cls._instances:
            instance = cls._instances[-1]
            instance.update_data(result, X, lens)

    def update_data(self, new_result, X=None, lens=None):
        """Updates mapper results and associated data, clearing cached visualizations"""
        self.result = new_result
        if X is not None:
            self.X = X
        if lens is not None:
            self.lens = lens
        self._data = None

    def _prepare_d3_data(self) -> dict:
        """
        Transforms mapper output into D3-compatible format.

        Returns
        -------
            dict: Network data structure for D3 visualization
        """
        if self._data is not None:
            return self._data

        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(len(self.result.nodes)))
        G.add_edges_from(self.result.nerve[1])
        positions = nx.spring_layout(G, dim=2, seed=42)
        positions = {k: v * 960 for k, v in positions.items()}

        nodes = []
        for i, members in enumerate(self.result.nodes):
            x, y = positions[i]
            nodes.append({
                "id": str(i),
                "x": float(x),
                "y": float(y),
                "size": len(members),
                "color": float(np.nanmean(self.result.projection[members])
                         if len(members) > 0 else 0.0),
                "tooltip": self._format_tooltip(members)
            })

        links = [{"source": str(s[0]), "target": str(s[1])}
                 for s in self.result.nerve[1]]

        faces = [{"nodes": [str(n) for n in simplex]}
                for simplex in self.result.nerve[2]]

        self._data = {
            "nodes": nodes,
            "links": links,
            "faces": faces
        }
        return self._data

    def _format_tooltip(self, members: np.ndarray) -> str:
        """
        Generates HTML tooltip content for node hover interactions.

        Because everyone loves a good tooltip! 🎈
        """
        if members.size == 0:
            return "Empty cluster"

        tooltip = [f"<strong>Members:</strong> {len(members)}<br>"]

        if self.X is not None:
            cluster_data = self.X[members]
            means = np.mean(cluster_data, axis=0)
            stds = np.std(cluster_data, axis=0)

            for i, (mean, std) in enumerate(zip(means, stds)):
                name = self.X_names[i] if i < len(self.X_names) else f"Feature {i}"
                tooltip.append("{name}:\n    {mean:>8.2f} ± {std:<8.2f}<br>".format(
                    name=name,
                    mean=mean,
                    std=std
                ))

        return "".join(tooltip)

    def show(self, port: int = 5000, debug: bool = True) -> None:
        """
        Launch interactive visualization in a web browser.

        Sets up a Flask server with live-reload capabilities.
        Warning: May cause excessive browser tab accumulation.

        Args:
            port: Server port number (default: 5000)
            debug: Enable Flask debug mode (default: True)
        """
        global app
        app = Flask(__name__)
        app.changed = False
        app.new_mapper_result = None

        # get the main script file path
        import __main__
        main_file = os.path.abspath(__main__.__file__) if hasattr(__main__, '__file__') else None

        if main_file:
            # set up file watching
            event_handler = FileChangeHandler(app, main_file)
            observer = Observer()
            observer.schedule(event_handler, path=os.path.dirname(main_file), recursive=False)
            observer.start()
            print(f"Watching for changes in: {os.path.dirname(main_file)}")
        else:
            print("Warning: Could not determine main file path. Auto-update disabled.")

        @app.route('/')
        def index():
            return render_template('base.html',
                                title='Mapper Visualization',
                                mapper_data=json.dumps(self._prepare_d3_data(), cls=NumpyEncoder))

        @app.route('/get_updated_data')
        def get_updated_data():
            if app.new_mapper_result is not None:
                self.update_data(app.new_mapper_result)
                app.new_mapper_result = None
            return json.dumps(self._prepare_d3_data(), cls=NumpyEncoder)

        @app.route('/stream')
        def stream():
            def generate():
                while True:
                    if app.changed:
                        print("Sending change event")
                        app.changed = False
                        yield "data: change detected\n\n"
                    time.sleep(0.5)
            response = Response(generate(), mimetype='text/event-stream')
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['X-Accel-Buffering'] = 'no'
            return response

        try:
            app.run(debug=debug, port=port, use_reloader=False, threaded=True)
        finally:
            if main_file:  # only try to stop observer if it was started
                observer.stop()
                observer.join()

    def render(self, output_file: str = "mapper.html") -> str:
        """
        Generate a static HTML visualization.

        For when you want to share your mapperpiece!

        Args
        ----
            output_file: Path for the output HTML file

        Returns
        -------
            str: Absolute path to the generated HTML file
        """
        current_dir = Path(__file__).parent

        env = Environment(
            loader=FileSystemLoader(current_dir / "templates"),
            trim_blocks=True,
            lstrip_blocks=True
        )

        template_path = current_dir / "templates" / "base.html"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found at {template_path}")

        template = env.get_template("base.html")

        html = template.render(
            title="Mapper Visualization",
            mapper_data=json.dumps(self._prepare_d3_data(), cls=NumpyEncoder)
        )

        output_path = Path(output_file)
        output_path.write_text(html)

        return str(output_path.resolve())

class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder with numpy array support:
    because numpy arrays need love too! 💝
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
