# AIN-2002: Introduction to Data Science — 2025/2026 Spring

This repository contains teaching materials, notebooks, practice files, datasets, homework, quizzes, and exam-related resources for **AIN-2002: Introduction to Data Science**.

The repository is designed as a course workspace for students who are learning practical data science with Python, NumPy, pandas, data wrangling, data cleaning, file-based data ingestion, exploratory data analysis, time-series handling, and data visualization.

## Repository Overview

The materials are organized around hands-on computational notebooks and supporting course resources.

| Path / File | Purpose |
|---|---|
| `python.basics.ipynb` | Introductory Python syntax and programming fundamentals. |
| `basic.data.structures.ipynb` | Core Python data structures such as lists, tuples, dictionaries, and sets. |
| `numpy.basics-part1.ipynb` | First part of NumPy fundamentals. |
| `numpy.basics-part2.ipynb` | Second part of NumPy fundamentals and extended numerical operations. |
| `pandas.basics.ipynb` | Introduction to pandas data structures and basic dataframe operations. |
| `data.loading.and.file.formats.ipynb` | Loading and working with structured data files. |
| `data.cleaning.and.preparation.ipynb` | Data quality, missing values, cleaning, and preparation workflows. |
| `data.wrangling.ipynb` | Data transformation and reshaping operations. |
| `data.aggregation.and.grouping.ipynb` | Grouping, aggregation, and summary-level analysis using pandas. |
| `time.series.ipynb` | Time-series data handling and analysis. |
| `visualization.ipynb` | Data visualization using Python libraries. |
| `practice.session-numpy.ipynb` | Guided NumPy practice session. |
| `practice.session-pandas.basics.ipynb` | Guided pandas practice session. |
| `sample_data/` | Sample datasets used in notebooks and exercises. |
| `exercises/` | Exercise documents for classroom or independent practice. |
| `homework/` | Homework assignments and supporting files. |
| `quizzes/` | Quiz source files organized by section. |
| `exams-2025.2026-spring/` | Exam materials for the current semester. |
| `exams-archive/` | Archived exam and mock exam materials from previous semesters. |
| `Requirements.txt` | Python package dependencies used by the repository. |
| `LICENSE` | MIT License information. |

## Learning Objectives

By working through this repository, students are expected to develop the ability to:

1. Use Python as a computational environment for introductory data science.
2. Represent, manipulate, and transform data using built-in Python data structures.
3. Apply NumPy for vectorized numerical computation.
4. Use pandas for tabular data processing, indexing, filtering, grouping, aggregation, and reshaping.
5. Load data from common file formats such as CSV, JSON, and Excel.
6. Identify and handle missing, inconsistent, or noisy data.
7. Conduct exploratory data analysis using systematic data wrangling workflows.
8. Work with time-indexed datasets and basic time-series operations.
9. Produce meaningful visualizations for data interpretation and communication.
10. Complete structured programming and data analysis tasks in notebook-based environments.

## Technical Stack

The repository primarily uses:

- **Python** for programming and data analysis
- **Jupyter Notebook** for interactive computation
- **NumPy** for numerical processing
- **pandas** for dataframe-based data analysis
- **Matplotlib**, **Seaborn**, and **Plotly** for visualization
- **OpenPyXL** for Excel file support
- **Requests** for HTTP-based data access examples

The repository is mostly composed of Jupyter Notebook files, with a small amount of Python script content.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/deepcloudlabs/ain2002-2025.2026-spring.git
cd ain2002-2025.2026-spring
```

### 2. Create a virtual environment

Using `venv`:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

The repository contains a `Requirements.txt` file listing the core packages. You can install the dependencies directly as follows:

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas mpmath requests matplotlib plotly nbformat openpyxl seaborn
```

If you want to run the notebooks locally, install JupyterLab as well:

```bash
python -m pip install jupyterlab
```

### 4. Start JupyterLab

```bash
jupyter lab
```

Then open the relevant `.ipynb` file from the JupyterLab interface.

## Recommended Study Path

A suggested progression for students is:

1. `python.basics.ipynb`
2. `basic.data.structures.ipynb`
3. `numpy.basics-part1.ipynb`
4. `numpy.basics-part2.ipynb`
5. `practice.session-numpy.ipynb`
6. `pandas.basics.ipynb`
7. `practice.session-pandas.basics.ipynb`
8. `data.loading.and.file.formats.ipynb`
9. `data.cleaning.and.preparation.ipynb`
10. `data.wrangling.ipynb`
11. `data.aggregation.and.grouping.ipynb`
12. `time.series.ipynb`
13. `visualization.ipynb`

This order follows a gradual transition from programming fundamentals to analytical data science workflows.

## Working with Sample Data

The `sample_data/` directory includes datasets in multiple formats, including CSV, JSON, and Excel. These files are intended to support practical examples on:

- data loading,
- file format comparison,
- tabular data analysis,
- time-series analysis,
- exploratory data analysis,
- data cleaning and transformation.

When using notebooks, make sure the notebook kernel is started from the repository root so that relative file paths resolve correctly.

## Homework, Quizzes, and Exams

The repository includes separate directories for assessment-related resources:

- `homework/` contains homework assignment documents and supporting files.
- `quizzes/` contains quiz scripts organized by section.
- `exams-2025.2026-spring/` contains exam material for the current semester.
- `exams-archive/` contains archived materials from earlier semesters.

Students should always follow the official course announcements, institutional assessment rules, and instructor-specific submission instructions. Repository content should not be interpreted as a replacement for official course communication.

## Academic Integrity

Students are expected to use this repository for learning, practice, and authorized coursework. Unless explicitly permitted by the instructor, students should not copy solutions, reuse another student's work, or submit externally generated answers as their own.

Appropriate use includes:

- studying notebooks,
- running and modifying examples,
- completing exercises independently,
- using sample datasets for practice,
- reviewing archived materials for preparation.

Inappropriate use includes:

- submitting copied code without understanding,
- sharing unauthorized solutions,
- using assessment materials in violation of course policy,
- misrepresenting generated or copied content as original work.

## Updating the Repository

If you are using this repository during the semester, pull the latest changes regularly:

```bash
git pull origin main
```

If you modified files locally, commit or back up your work before pulling updates.

## Troubleshooting

### Jupyter does not recognize the virtual environment

Install `ipykernel` and register the environment:

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name ain2002 --display-name "Python (AIN-2002)"
```

Then select **Python (AIN-2002)** as the notebook kernel.

### A package is missing

Install it using pip:

```bash
python -m pip install package-name
```

Then restart the notebook kernel.

### File not found errors occur in notebooks

Check that you launched Jupyter from the repository root directory:

```bash
cd ain2002-2025.2026-spring
jupyter lab
```

Relative paths such as `sample_data/...` are expected to resolve from the repository root.

## License

This repository is distributed under the **MIT License**. See the `LICENSE` file for details.

## Repository

GitHub repository: <https://github.com/deepcloudlabs/ain2002-2025.2026-spring>


