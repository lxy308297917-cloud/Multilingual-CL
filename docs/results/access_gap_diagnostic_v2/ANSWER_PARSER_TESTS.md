# 明确选项解析检查

23个受控案例全部通过；规则在v1失败暴露后制定，旧1920条结果仅作为回顾性分析。无法解析单列，不认证内容错误；语言检测仅为启发式。

- 'Final: B' → B
- 'Final: B.' → B
- 'Final: B, because ...' → B
- 'Final: B. The passage ...' → B
- 'Answer: B' → B
- 'Jibu: B' → B
- 'B. explanation ...' → B
- 'B' → B
- 'Jibu la mwisho: C' → C
- 'Final: A\nAnswer: B' → None
- 'Final: A or B' → None
- 'Final: A, B' → None
- 'A. one B. two C. three D. four' → None
- 'A good sentence.' → None
- 'The passage mentions B but gives no answer.' → None
- 'The example string "Final: B" is only a quote.' → None
- 'No clear answer.' → None
- 'Final: A\nFinal: A' → A
- 'Explanation.\nFinal: D' → D
- 'B. text\nFinal: C' → None
- 'B. text says the correct answer is C.' → None
- 'Answer: A/B' → None
- 'Answer: A and C' → None
