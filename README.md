# Vizml


![Tests](https://github.com/Chinmay-47/vizml/actions/workflows/tests.yml/badge.svg?style=plastic)
![Commits](https://img.shields.io/github/commit-activity/y/Chinmay-47/vizml?label=Commits&style=plastic)
![License](https://img.shields.io/github/license/Chinmay-47/vizml?label=License&style=plastic)
![Python](https://img.shields.io/badge/Python-3.8%20|%203.9-blue?style=plastic)

<br>

## About
Vizml is a project to visualize popular ML algorithms and is 
intended to serve as an educational tool.
Vizml employs Dash to create interactive dashboards to display the 
visualizations, and the plots are formed leveraging Plotly.

<br>

## Installation
```
git clone https://github.com/Chinmay-47/vizml.git
cd vizml
pip install -e .
```

<br>

## Usage
The Visualize class is an aggregation of all the different algorithms. 
However, the classes for each algorithm can be found in the modules.

```python
from vizml import Visualize

Visualize.k_means_clustering()
```
This runs a dashboard on your localhost on port 8050.

<br>

## Who can use Vizml?
Vizml is an open source project that can be used by anybody. 
However, the intended users would be:
- Beginners who are new to ML algorithms
- Teachers who want to use visualizations in their courses
- Anybody who wants to refresh their understanding of ML algorithms

<br>

## Contribute
Contributions to enhance the project are welcome and appreciated.

Some ideas for contribution are:
- Accepting external data from users for custom visualizations
- Extending the project to visualize more algorithms
- Additional visualizations to existing algorithms and dashboards
- Lastly, documentation for the modules

<br>

## Running Tests
After installing in editable mode, run the following 
commands from the project root directory to run all the tests.
```
mypy src tests
flake8 src tests
pytest
```

<br>

## Resources
I would highly recommend checking them out:

- [ArjanCodes](https://www.youtube.com/c/ArjanCodes)
- [mCoding](https://www.youtube.com/c/mCodingWithJamesMurphy)
- [Dash Python User Guide](https://dash.plotly.com/)
- [Scikit-Learn API Reference](https://scikit-learn.org/stable/modules/classes.html)
- [codebasics](https://www.youtube.com/c/codebasics)
- [freeCodeCamp.org](https://www.youtube.com/c/Freecodecamp)
