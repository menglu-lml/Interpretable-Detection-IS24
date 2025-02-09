# Interpretable_Detection
Code for Interspeech2024 Paper "Interpretable Temporal Class Activation Representation for Audio Spoofing Detection"

Author: Menglu Li, Xiao-Ping Zhang

Paper Link: https://arxiv.org/pdf/2406.08825

## Abstract
Explaining the decisions made by audio spoofing detection models is crucial for fostering trust in detection outcomes. However, current research on the interpretability of detection models is limited to applying XAI tools to post-trained models. In this paper, we utilize the wav2vec 2.0 model and attentive utterance-level features to integrate interpretability directly into the model's architecture, thereby enhancing transparency of the decision-making process. Specifically, we propose a class activation representation to localize the discriminative frames contributing to detection. Furthermore, we demonstrate that multi-label training based on spoofing types, rather than binary labels as bonafide and spoofed, enables the model to learn distinct characteristics of different attacks, significantly improving detection performance. Our model achieves state-of-the-art results, with an EER of 0.51\% and a min t-DCF of 0.0165 on the ASVspoof2019-LA set.

## Framework
<p align='center'>  
<img src='https://github.com/menglu-lml/Interpretable_Detection_Interspeech24/blob/main/img/overview.png' width='870'/>
</p>

## Result
<p align='center'>  
<img src='https://github.com/menglu-lml/Interpretable_Detection_Interspeech24/blob/main/img/result.png' width='870'/>
</p>

## Training
The followng command runs the training and validation experiment.
```
python training.py --database_path="path/to/the/directory/of/ASVSPOOF2019/LA/database" --protocols_path="path/to/the/directory/of/ASVSPOOF2019/LA/protocols"
```

The default configurations are saved at `model_config.yaml`. If you would like change the configurations of the model, simply change the values in this file directly.

## Contact
If you have questions, please contact `menglu.li@torontomu.ca`.
## Citation
Comming Soon
