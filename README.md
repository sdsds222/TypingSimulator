# TypingSimulator
一个基于 Transformer Seq2Seq 模型的打字行为模拟器，能够预测并模拟人类用户在输入目标文本时所采取的详细动作和时间节奏，包括拼音输入、汉字/英文确认，以及删除修正等所有步骤。It simulates the detailed actions and timing a human user takes to type a specific target text, including Pinyin input, character/word confirmation, and deletions.

一个基于 **Transformer Seq2Seq 模型** 的 **打字行为模拟器**。

它的核心功能是：预测并模拟人类用户在输入目标文本时所采取的详细动作和时间节奏，包括拼音输入、汉字/英文确认，以及删除修正等所有步骤。

该交互式脚本独特地实现了 **逐字驱动的实时模拟**：
1.  模型为目标文本中的 *每个字符* 分别生成一系列打字动作（例如：拼音按键、确认输入）。
2.  屏幕输出动态更新，实时展示 **拼音被确认的汉字瞬间替换** 的过程，高度还原真实的输入法体验。

收集输入行为数据集：
` python input.py `

浏览器访问：http://localhost:5000/

训练：
` python main.py `

脚本会读取当前目录下的名为：typing_data_with_pinyin.json 的数据集，训练完成之后在当前目录生成一个名为： typing_model.pth 的模型参数文件。

交互式访问：
` py .\interactive.py .\typing_model.pth `

参数是模型文件。

程序使用Claude辅助编写。

This Python program is a **Typing Behavior Simulator** based on a **Transformer Seq2Seq model**.

It simulates the detailed actions and timing a human user takes to type a specific target text, including Pinyin input, character/word confirmation, and deletions.

The interactive script is specifically designed to perform **character-by-character, real-time simulation**, where:
1.  The model generates a sequence of actions (Pinyin keys, Type actions) for *each character* in the target text.
2.  The screen output is dynamically updated, showing the Pinyin being **replaced** instantly by the confirmed Chinese character, mimicking a realistic IME (Input Method Editor) experience.

