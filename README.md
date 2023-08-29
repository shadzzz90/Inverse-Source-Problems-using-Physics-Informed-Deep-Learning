# Inverse-Source-Problems-using-Physics-Informed-Deep-Learning


## Overview

This repository presents a novel deep learning-based approach to tackle the _Inverse Source Problem_ (ISP) in engineering design. When a system's dynamic response is known or measurable but the source generating these responses remains unknow, it poses a challenge. Solving this challenge is critical, and while traditional techniques rely on optimization and regularization, we've taken a deep dive into the realm of deep learning.

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
   ```bash
   python main.py --input path_to_input_data --output path_to_save_results
   ```

## Future Work

While the current results are promising, the domain is vast, and there's more to explore. Future enhancements include:
- Extending the model to systems with multiple degrees of freedom.
- Incorporating real-world noise and uncertainties in the data.

## Contributing

Contributions are welcomed! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to proceed.

## Citation

If you use this work or repository in your research or project, please cite our paper:

```bibtex
@article{authors2023deepinverse,
  title={Recovering the Forcing Function in Systems with One Degree of Freedom Using ANN and Physics Information},
  author={Shadab Anwar Shaikh , Harish Cherukuri, and Taufiquar Khan},
  journal={Algorithms},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

For any inquiries, please open an issue or reach out to the main authors via [email@example.com](mailto:email@example.com).

---

Proudly powered by passionate researchers.
