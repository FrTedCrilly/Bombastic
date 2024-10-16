import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO
from flask import Flask, render_template_string
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)


class DataPlotter:
    """
    A class to plot time series and performance statistics and return the plots as HTML.
    """

    def __init__(self, data_dict):
        """
        Initialize the DataPlotter with a data dictionary.

        Parameters:
        data_dict (dict): Nested dictionary containing the data to be plotted.
        """
        self.data_dict = data_dict
        self.plots_html = self.plot_to_html()

    def create_app(self):
        """
        Create a Flask app and add the route for the index page.
        """
        app = Flask(__name__)

        # Use a closure to bind the method to the route
        @app.route('/')
        def index():
            return self.index()

        return app

    def plot_to_html(self):
        """
        Plots time series and performance statistics for each key in the nested data dictionary
        and returns the plots and stats table in HTML format.
        """
        logging.debug("Starting plot_to_html")
        plots_html = ""
        for outer_key, inner_dict in self.data_dict.items():
            for key, value in inner_dict.items():
                logging.debug(f"Processing {outer_key} - {key}")
                # Create a new figure
                fig, axs = plt.subplots(2, 1, figsize=(10, 10))

                # Plot the time series
                axs[0].plot(value[0], label='Cumulative Returns')
                axs[0].set_title(f'Cumulative Returns for {outer_key} - {key}')
                axs[0].legend()

                # Prepare the stats table
                stats = value[1]
                stats_names = ['SR', 'Sortio', 'skew', 'kurt', 'valAdd', 'refAdd', 'retCorr']
                stats_values = [
                    stats[stat].values[0] if isinstance(stats[stat], (pd.Series, pd.DataFrame)) else stats[stat] for
                    stat in stats_names]

                # Create a table for the stats
                cell_text = [[f'{val:.4f}' if isinstance(val, float) else str(val)] for val in stats_values]
                table = axs[1].table(cellText=cell_text,
                                     rowLabels=stats_names,
                                     colLabels=['Value'],
                                     loc='center',
                                     cellLoc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.5, 1.5)
                for cell in table.get_celld().values():
                    cell.set_edgecolor('black')
                    cell.set_linewidth(0.5)
                axs[1].axis('off')
                axs[1].set_title(f'Stats for {outer_key} - {key}')

                # Add TAnalysis details if present
                if 'TAnalysis' in stats:
                    tanalysis = stats['TAnalysis']
                    ta_names = list(tanalysis.keys())
                    ta_values = [
                        tanalysis[ta].values[0] if isinstance(tanalysis[ta], (pd.Series, pd.DataFrame)) else tanalysis[
                            ta] for ta in ta_names]
                    ta_text = [[f'{val:.4f}' if isinstance(val, float) else str(val)] for val in ta_values]
                    ta_table = axs[1].table(cellText=ta_text,
                                            rowLabels=ta_names,
                                            colLabels=['Value'],
                                            loc='bottom',
                                            cellLoc='center')
                    ta_table.auto_set_font_size(False)
                    ta_table.set_fontsize(10)
                    ta_table.scale(1.5, 1.5)
                    for cell in ta_table.get_celld().values():
                        cell.set_edgecolor('black')
                        cell.set_linewidth(0.5)

                # Save the plot to a PNG image in memory
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)

                # Add the image and stats table to the HTML string
                plots_html += f"""
                <h2>{outer_key} - {key}</h2>
                <img src="data:image/png;base64,{image_base64}" />
                """

        logging.debug("Finished plot_to_html")
        return plots_html

    def index(self):
        """
        Serve the HTML content.
        """
        logging.debug("Serving index page")
        html_template = f"""
        <html>
        <head><title>Plot and Stats</title></head>
        <body>
        {self.plots_html}
        </body>
        </html>
        """
        return render_template_string(html_template)

    def run_flask(self):
        """
        Run the Flask web server.
        """
        logging.debug("Starting Flask app")
        app = self.create_app()
        app.run(debug=True)


if False:
    example_dict = {
        'Strategy1': {
            'BBands_20_window_1_sd_num_True_reversal': [
                [0, 1, 2, 3, 4],  # Example time series data
                {
                    'SR': pd.Series([0.202891], index=['BBands_20_window_1_sd_num_True_reversal']),
                    'Sortio': pd.Series([0.289192], index=['BBands_20_window_1_sd_num_True_reversal']),
                    'skew': pd.Series([-0.062682], index=['BBands_20_window_1_sd_num_True_reversal']),
                    'kurt': pd.Series([3.517235], index=['BBands_20_window_1_sd_num_True_reversal']),
                    'valAdd': 0,
                    'refAdd': 0,
                    'retCorr': 0,
                    'TAnalysis': {
                        'Long Returns': pd.Series([4.649543], index=['BBands_20_window_1_sd_num_True_reversal']),
                        'Short Returns': pd.Series([-2.872506], index=['BBands_20_window_1_sd_num_True_reversal']),
                        'Proportion Long to Short': 0.37863182167563414,
                        'Hit Rate': 0.49177555726364336
                    }
                }
            ]
        }
    }
