
<a name="readme-top"></a>




<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">Detecting Lies in Text</h3>

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->

## About The Project

Lie Detection in text is a very active research field. From new stories to tweets interest in detecting lies is constantly growing. 

Many different datasets for training and many different models to do so exist. 

This project explores how these models perform within specific contexts and their generalisation  across datasets. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python.com]][python-url]
* [![PyTorch][PyTorch.com]][PyTorch-url]
* [![scikit-learn][scikit-learn.com]][scikit-learn-url]
* [![Matplotlib][Matplotlib.com]][Matplotlib-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- PREREQUISITES -->
## Prerequisites

It requires Python 3.6+, PyTorch 1.5+, and preferably, CUDA 9.2+.

<!-- USAGE EXAMPLES -->
## Directory Structure
```
.
├── Basic models
│   ├── src
│   │   ├── *.ipynb             # Code of basic models
│   ├── datasets                # Including raw data and pre-processed data
│   ├── toolkit
│   │   ├── sentiment_lexicon   # Sentiment feature generator
│   │   └── ark-tweet-nlp-0.3.2 # Part-of-speech feature generator

├── Deep learning models
│   ├── main_lstm.py            # LSTM program entrance
│   ├── main_bert.py            # BERT program entrance
│   ├── main_trans.py           # BERT of transfer learning entrance
│   ├── src
│   │   ├── *.py                # Code of deep learning models
│   ├── datasets                # pre-processed data
│   ├── log                     # Logs of training processes
│   └── save                    # Saved models
└── README.md
```

## Usage
This part tells readers how to reproduce the research process of the project.

For basic models, just run the cells of *.ipynb files stepwise. To be more specific, the chronological running order of the files is 'Data Preprocessing and Feature Engineering.ipynb', 'Modelling.ipynb', and 'data_analysis.ipynb'.

For deep learning models, execute the following command line format.
* Format:  
  ```
  python <python_file_name>.py > '>' <log_file_name>.out --fgm <and/or> --ema <and/or> --aug <and/or> ...
  ```  
  Optional arguements '--fgm', '--ema', '--aug' indicate the optional optimisation methods of the models.
* Examples:
    1. ```python main_lstm.py > lstm_fgm.log --fgm```
    2. ```python main_bert.py > bert_ema_aug.log --ema --aug```
    3. ```python main_trans.py > trans_fgm_aug.log --fgm --aug```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Publication
[Link](https://pats.cs.cf.ac.uk/!archive_desc?p=2572)

## Contact

Qin Liu - [LinkedIn](https://www.linkedin.com/in/qin-liu-b100501ab/) - qin.liu01@estudiant.upf.edu / qinliu1996@gmail.com

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

I highly appreciate [Prof. Yukun Lai]() and [Dr. Krill Sidorov](https://users.cs.cf.ac.uk/K.Sidorov/), who have instructed me quiet a lot throughout the research of the project.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[Python.com]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[PyTorch.com]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[PyTorch-url]: https://pytorch.org/
[scikit-learn.com]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[scikit-learn-url]: https://scikit-learn.org/stable/
[Matplotlib.com]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]: https://matplotlib.org/
