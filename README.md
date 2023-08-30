# Inverse-Source-Problems-using-Physics-Informed-Deep-Learning


## Overview

This repository presents a novel deep learning-based approach to tackle the Inverse Source Problem (ISP) in engineering design. In variety of engineering problems system's dynamic response is known or measurable but the source generating these responses is not known. Addressing this issue is critical, and while traditional techniques rely on optimization and regularization, we've taken a deep dive into the realm of deep learning.


## Key Features

- Deep learning architecture crafted explicitly for the ISP.
- Incorporates physics-based information to enhance recovery accuracy.
- Capable of recovering:
  - Smooth forcing functions.
  - Abrupt gradient changes.
  - Jump discontinuities for linear systems.
  - Harmonics and their combinations for non-linear systems.

## Motivation

Inverse source problems are quintessential in engineering, particularly when deriving the forces or inputs from observed responses. Traditionally, ISPs are tackled using optimization techniques with regularization. However, with the advances in deep learning, there's a growing interest to leverage it for more accurate and efficient solutions. We address this by offering a physics-infused deep learning method.

## Results

Our approach has shown considerable success:
- Efficient recovery of smooth forcing functions.
- Handling abrupt changes and jump discontinuities in linear systems.
- Accurately recovering harmonic functions, their summations, and Gaussian functions in non-linear systems.

The results underline the potential of combining deep learning with physics for solving complex engineering problems.

## Getting Started

1. **Prerequisites**:
   - Ensure you have [Python 3.8](https://www.python.org/downloads/) or later installed.
   - Install required packages: `pip install -r requirements.txt`

2. **Usage**:
   - Training:
     ```bash
     python Train.py
     ```
   - Testing:
     ```bash
      python Test.py
     ```

## Future Work

While the current results are promising, the domain is vast, and there's more to explore. Future enhancements include:
- Extending the model to systems with multiple degrees of freedom.
- Incorporating real-world noise and uncertainties in the data.


## Citation

If you use this work or repository in your research or project, please cite our paper:

```bibtex
@article{SAShaikh2023deepinverse,
  title={Recovering the Forcing Function in Systems with One Degree of Freedom Using ANN and Physics Information},
  author={Shadab Anwar Shaikh , Harish Cherukuri, and Taufiquar Khan},
  journal={Algorithms},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

For any inquiries, please open an issue or reach out to the main authors via [sshaikh4@charlotte.edu](mailto:sshaikh4@charlotte.edu).

---

Proudly powered by passionate researchers.
