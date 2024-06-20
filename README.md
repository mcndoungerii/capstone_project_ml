[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/duarteocarmo/interactive-dashboard-post/master?urlpath=%2Fvoila%2Frender%2Fnotebooks%2FDashboard.ipynb)  [![Python](https://img.shields.io/badge/python-v3.9-blue)](https://www.python.org/)  [![saythanks](https://img.shields.io/badge/say-thanks-ff69b4.svg)](https://duarteocarmo.com)

# From notebook to web application 📔​+🔮=💥 

## How do I run the notebook? :notebook_with_decorative_cover:

Clone the repo:

```bash
$ git remote add origin https://github.com/mcndoungerii/capstone_project_ml.git
```

Navigate to it:

```bash
$ cd interactive-dashboard-post
```

Create a [virtual environment](https://virtualenv.pypa.io/en/latest/):

```bash
$ virtualenv env
```

Activate the environment:

```bash
$ . env/bin/activate
```

Install the requirements:

```bash
pip install -r requirements.txt 
```

Launch [jupyter lab](https://jupyterlab.readthedocs.io/en/stable/):

```bash
jupyter lab
```

It should launch automatically. If not, [check this](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html).

🚨 **troubleshooting** 🚨

If the plotly express plots are not showing then try:

```bash
jupyter labextension install @jupyterlab/plotly-extension
```

If you still have problems, follow [these instructions](https://plot.ly/python/getting-started/#jupyterlab-support-python-35).



## How do I run the dashboard? :bar_chart:

Follow the instructions above until you have the requirements installed, and then: 

```bash
voila notebooks/malicious_benign.ipynb
```

This should launch the dashboard in http://localhost:8866/
