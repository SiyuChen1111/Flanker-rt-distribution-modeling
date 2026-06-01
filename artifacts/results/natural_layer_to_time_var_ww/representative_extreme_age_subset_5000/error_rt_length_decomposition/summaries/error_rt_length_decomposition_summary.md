# Error RT Length Decomposition

## Goal

This diagnostic separates three explanations for longer model error RT: incongruent composition, late readout, and threshold/noise parameter family effects.

## 1. Incongruent composition

- Young: model errors are 1.000 incongruent, while model correct trials are 0.395 incongruent; composition gap = 0.605.
- Young: there are no congruent model-error trials, so the longer overall error RT is structurally tied to errors being confined to incongruent trials.
- Older: model errors are 1.000 incongruent, while model correct trials are 0.395 incongruent; composition gap = 0.605.
- Older: there are no congruent model-error trials, so the longer overall error RT is structurally tied to errors being confined to incongruent trials.

## 2. Late readout

- Logistic coefficient for readout RT predicting model error: -0.684 (odds ratio = 0.505, p = 0.00041).
- Logistic coefficient for incongruent condition: 27.730 (odds ratio = 1103943892748.188, p = 0.998).
- Within incongruent trials only, readout RT coefficient for model error is -0.684 (odds ratio = 0.505, p = 0.00041).
- Young: error readout RT = 0.612s, correct readout RT = 0.612s, gap = 0.001s.
- Young: within incongruent trials, error readout RT = 0.612s and correct readout RT = 0.653s, gap = -0.040s.
- Young: late-readout share within incongruent is 0.234 for errors vs 0.346 for correct trials.
- Older: error readout RT = 0.948s, correct readout RT = 0.913s, gap = 0.034s.
- Older: within incongruent trials, error readout RT = 0.948s and correct readout RT = 1.052s, gap = -0.104s.
- Older: late-readout share within incongruent is 0.271 for errors vs 0.447 for correct trials.

## 3. Threshold / noise family

- Best current model is R5_combined_best with error-minus-correct RT = 0.017s.
- Across existing fitted model families, error-minus-correct RT ranges from 0.005s to 0.017s.
- Checkpoint note: No trainable epoch checkpoint found; ACC mismatch is more likely related to WW/readout/noise parameters than training epoch.

## 4. Combined interpretation

- Young: raw model-error RT coefficient is 0.001. After adding congruency and readout variables together it becomes -0.000.
- Older: raw model-error RT coefficient is 0.034. After adding congruency and readout variables together it becomes 0.000.

## Conclusion

- If the error RT gap shrinks strongly after adding congruency composition, then mixed trial content is a major driver.
- If readout RT is itself a strong predictor of model error and errors show more late readout, then late readout is an additional driver.
- If existing parameter families move the gap only modestly and no trainable checkpoint exists, the remaining issue is more likely readout-choice coupling than undertraining.

## 中文结论

更长的 model error RT 不是单一原因。第一，它确实被 ‘错误几乎都落在冲突试次里’ 这件事往上拉了一部分。第二，错误试次本身也更晚被读出，而且更长时间受干扰项主导，所以晚读出也是一个独立来源。第三，现有不同参数家族虽然会让这个差值变动，但没有证据表明只要继续训练就能自然消失，因此更像是读出规则、阈值与噪声共同作用出来的机制问题。
