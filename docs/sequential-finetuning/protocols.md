# 冻结测评协议与完整提示

[总览](README.md) · [阶段矩阵](stage-matrices.csv) · [单语结果](single-and-baselines.md)

这些提示用于固定 Qwen3-4B Base 及训练后模型。花括号为题目字段，助手从模板中标出的前缀之后作答。每个基准在各自划分内使用固定题号、解码和评分。主要训练预算、分组与留出题量见总览。

## XNLI

使用 `facebook/xnli` 固定版本 `b8dd5d7af51114dbda02c0e3f6133f332186418e`。每语官测 5,010 题，按覆盖三类的英语前提组配对重采样。含英语原序用统一英文提示。训练样本源题及规范化前提组跨训练语言互斥，与标准开发和测试前提排重。无英语对照每语 2,000 条，固定相同源题、标签、监督词元及顺序。语言化提示会改变输入词元数。无英语两个训练指令组均使用本地推理指令测评。下方是原始输入，三个候选 ` entailment`、` neutral`、` contradiction` 各在 `Relation:` 后续接，均有前导空格。固定 lm-evaluation-harness 端到端计算似然，候选并列取列出的首项。分母为固定题数。 [XNLI 原论文](https://aclanthology.org/D18-1269/)

```text
<|im_start|>user
Given the premise and hypothesis, classify their relation. Choose one label: entailment, neutral, or contradiction.

Premise: {premise}
Hypothesis: {hypothesis}<|im_end|>
<|im_start|>assistant
<think>

</think>

Relation:
```

无英语试训与八语预筛用下列首行替换同一模板的英文首行。每项语言代码只用来标记对应首行。

```text
en
Given the premise and hypothesis, classify their relation. Choose one label: entailment, neutral, or contradiction.
de
Klassifiziere die Beziehung zwischen der Prämisse und der Hypothese. Wähle genau eines dieser Labels: entailment, neutral oder contradiction.
ar
صنّف العلاقة بين المقدّمة والفرضية. اختر تصنيفًا واحدًا: entailment أو neutral أو contradiction.
vi
Phân loại mối quan hệ giữa tiền đề và giả thuyết. Chọn đúng một nhãn: entailment, neutral hoặc contradiction.
sw
Ainisha uhusiano kati ya kauli ya msingi na dhana. Chagua lebo moja: entailment, neutral au contradiction.
ur
مقدمے اور مفروضے کے درمیان تعلق کی درجہ بندی کریں۔ صرف ایک لیبل منتخب کریں: entailment، neutral یا contradiction۔
hi
आधार कथन और परिकल्पना के बीच संबंध का वर्गीकरण करें। ठीक एक लेबल चुनें: entailment, neutral या contradiction।
th
จำแนกความสัมพันธ์ระหว่างข้อความตั้งต้นกับสมมติฐาน เลือกป้ายกำกับเพียงหนึ่งรายการ: entailment, neutral หรือ contradiction
el
Ταξινόμησε τη σχέση ανάμεσα στην προκείμενη και την υπόθεση. Επίλεξε ακριβώς μία ετικέτα: entailment, neutral ή contradiction.
```

旧含英语模型在官方开发集上的本地推理复测另用相同模板替换首行。德语、阿语、SW 和中文实际首行如下。

```text
en
Given the premise and hypothesis, classify their relation. Choose one label: entailment, neutral, or contradiction.
de
Klassifiziere die Beziehung zwischen der Prämisse und der Hypothese. Wähle genau eines dieser Labels: entailment, neutral oder contradiction.
ar
صنّف العلاقة بين المقدّمة والفرضية. اختر تصنيفًا واحدًا: entailment أو neutral أو contradiction.
sw
Ainisha uhusiano kati ya kauli ya msingi na dhana. Chagua lebo moja: entailment, neutral au contradiction.
zh
判断前提和假设之间的关系。请选择一个标签：entailment、neutral 或 contradiction。
```

## MASSIVE 1.1

五语同版官方语料，官方开发每语 2,033 题，测试每语 2,974 题。输入使用作者词序列。回放按输入与监督词元约束匹配替换，原顺序和换序分别保持同一批训练样本。槽位用作者词边界计分，边界和类别均正确才计命中，实调作者 BIO 转换与 seqeval 1.2.2。主指标是按该语言全量题累计 TP、FP、FN 得到的 `2TP/(2TP+FP+FN)`。同时按全题报告意图准确率、整句匹配和结构有效率。贪心，至多生成 192 词元。冻结的槽位示例、完整标签表如下。 [MASSIVE 作者评分器](https://github.com/alexa/massive/tree/f966f21846043aabef9b0f974fa7970027f43738)

```text
<|im_start|>user
Identify the intent and extract the slots of the utterance. Return one JSON object with exactly two keys: "intent" and "slots". Use one canonical intent from the list. Each slot is [canonical slot type, exact text copied from the utterance, occurrence]. Occurrence starts at 1 and counts exact word-sequence matches in the utterance from left to right. Occurrence is counted separately for each exact text, never as the slot number. If two different texts each appear once, both slots have occurrence 1. Output the slots in utterance order. Use an empty list when no slot applies.
Example utterance: set an alarm for tomorrow at seven
Example JSON: {"intent":"alarm_set","slots":[["date","tomorrow",1],["time","seven",1]]}
Canonical intents: alarm_query, alarm_remove, alarm_set, audio_volume_down, audio_volume_mute, audio_volume_other, audio_volume_up, calendar_query, calendar_remove, calendar_set, cooking_query, cooking_recipe, datetime_convert, datetime_query, email_addcontact, email_query, email_querycontact, email_sendemail, general_greet, general_joke, general_quirky, iot_cleaning, iot_coffee, iot_hue_lightchange, iot_hue_lightdim, iot_hue_lightoff, iot_hue_lighton, iot_hue_lightup, iot_wemo_off, iot_wemo_on, lists_createoradd, lists_query, lists_remove, music_dislikeness, music_likeness, music_query, music_settings, news_query, play_audiobook, play_game, play_music, play_podcasts, play_radio, qa_currency, qa_definition, qa_factoid, qa_maths, qa_stock, recommendation_events, recommendation_locations, recommendation_movies, social_post, social_query, takeaway_order, takeaway_query, transport_query, transport_taxi, transport_ticket, transport_traffic, weather_query
Canonical slot types: alarm_type, app_name, artist_name, audiobook_author, audiobook_name, business_name, business_type, change_amount, coffee_type, color_type, cooking_type, currency_name, date, definition_word, device_type, drink_type, email_address, email_folder, event_name, food_type, game_name, game_type, general_frequency, house_place, ingredient, joke_type, list_name, meal_type, media_type, movie_name, movie_type, music_album, music_descriptor, music_genre, news_topic, order_type, person, personal_info, place_name, player_setting, playlist_name, podcast_descriptor, podcast_name, radio_name, relation, song_name, sport_type, time, time_zone, timeofday, transport_agency, transport_descriptor, transport_name, transport_type, weather_descriptor
Utterance: {utterance}<|im_end|>
<|im_start|>assistant
<think>

</think>

JSON:
```

## MasakhaNER 2.0

同版四语官方语料，跨语检查训练句与开发和测试文本的重叠，开发每语固定子集 384。以官方完整测试的 BIO 标注做实体级微 F1。用官方分词编号，解析生成跨度后映射到 BIO，非法结构以空预测计分，边界和标签同时正确才命中。vLLM 0.9.2 V0，贪心，最多 160 新词元。 [MasakhaNER 固定版本](https://github.com/masakhane-io/masakhane-ner/tree/ba5843cd08aa491d5f96a5e809e71eb9ec461391/MasakhaNER2.0)

```text
<|im_start|>user
Identify every named entity in this numbered token sequence. Use PER for a person, ORG for an organization, LOC for a location, and DATE for a date or time expression. Return only a JSON list of [start,end,"TYPE"] triples. Token numbering starts at 0. The start is included and the end is excluded. Return [] if there is no entity.

Tokens:
{words}<|im_end|>
<|im_start|>assistant
<think>

</think>

Entities:
```

## TyDiQA-GoldP

每题输入金标段落和问题。内部开发取源训练侧每语 200 题，作者开发与训练按题号、标题和篇章去重。完整作者开发使用 7,936 上下文窗口，内部开发使用 1,536，贪心生成至多 96 词元。对输出去首尾空白，调用 SQuAD 1.1 归一化。每题在多个金标中分别选择最佳 EM 和词元 F1，按该语全题平均。配对区间按标题整组重采样。 [上游 SQuAD 1.1 评分器](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

```text
<|im_start|>user
Read the passage and answer the question. Copy only the shortest text span from the passage that answers the question.

Passage: {passage}

Question: {question}<|im_end|>
<|im_start|>assistant
<think>

</think>

Answer:
```

## MTOP

原始多语源包的 SHA-256 为 `0c086500aed2ed48d3df383c91664b152e4a46e2adef6a635fa16d508ebaec0b`。五个训练集从共有有效源 ID 中抽取，单语 1,024 条、各阶段源 ID 互斥。全量使用对应语言的开发和测试。顶层平面槽位评分要求类型和两端边界正确，按语累计混淆计数并用 seqeval 1.2.2 交叉检查。生成使用 vLLM 0.9.2 V0，贪心，2,048 上下文与 160 新词元。生成无效时按空槽位处理，意图可独立评分。附报词级 BIO 标签宏 F1。 [MTOP 原论文](https://aclanthology.org/2021.eacl-main.257/) · [作者数据](https://dl.fbaipublicfiles.com/mtop/mtop.zip)

```text
<|im_start|>user
Classify the utterance intent and extract the top-level slots. Return a JSON object with keys "intent" and "slots". Each slot is [start,end,"TYPE"], with zero-based token positions, inclusive start and exclusive end. Use the canonical labels exactly. Output slots in token order. Use [] when there is no slot.
Example tokens: 0:play 1:some 2:jazz
Example JSON: {"intent":"IN:PLAY_MUSIC","slots":[[2,3,"SL:MUSIC_GENRE"]]}
Canonical intents: IN:ADD_TIME_TIMER, IN:ADD_TO_PLAYLIST_MUSIC, IN:ANSWER_CALL, IN:CANCEL_CALL, IN:CANCEL_MESSAGE, IN:CREATE_ALARM, IN:CREATE_CALL, IN:CREATE_PLAYLIST_MUSIC, IN:CREATE_REMINDER, IN:CREATE_TIMER, IN:DELETE_ALARM, IN:DELETE_PLAYLIST_MUSIC, IN:DELETE_REMINDER, IN:DELETE_TIMER, IN:DISLIKE_MUSIC, IN:DISPREFER, IN:END_CALL, IN:FAST_FORWARD_MUSIC, IN:FOLLOW_MUSIC, IN:GET_AGE, IN:GET_AIRQUALITY, IN:GET_ALARM, IN:GET_ATTENDEE_EVENT, IN:GET_AVAILABILITY, IN:GET_CALL, IN:GET_CALL_CONTACT, IN:GET_CALL_TIME, IN:GET_CATEGORY_EVENT, IN:GET_CONTACT, IN:GET_CONTACT_METHOD, IN:GET_DATE_TIME_EVENT, IN:GET_DETAILS_NEWS, IN:GET_EDUCATION_DEGREE, IN:GET_EDUCATION_TIME, IN:GET_EMPLOYER, IN:GET_EMPLOYMENT_TIME, IN:GET_EVENT, IN:GET_GENDER, IN:GET_GROUP, IN:GET_INFO_CONTACT, IN:GET_INFO_RECIPES, IN:GET_JOB, IN:GET_LANGUAGE, IN:GET_LIFE_EVENT, IN:GET_LIFE_EVENT_TIME, IN:GET_LOCATION, IN:GET_LYRICS_MUSIC, IN:GET_MAJOR, IN:GET_MESSAGE, IN:GET_MESSAGE_CONTACT, IN:GET_MUTUAL_FRIENDS, IN:GET_RECIPES, IN:GET_REMINDER, IN:GET_REMINDER_AMOUNT, IN:GET_REMINDER_DATE_TIME, IN:GET_REMINDER_LOCATION, IN:GET_STORIES_NEWS, IN:GET_SUNRISE, IN:GET_SUNSET, IN:GET_TIMER, IN:GET_TRACK_INFO_MUSIC, IN:GET_UNDERGRAD, IN:GET_WEATHER, IN:HELP_REMINDER, IN:HOLD_CALL, IN:IGNORE_CALL, IN:IS_TRUE_RECIPES, IN:LIKE_MUSIC, IN:LOOP_MUSIC, IN:MERGE_CALL, IN:PAUSE_MUSIC, IN:PAUSE_TIMER, IN:PLAY_MEDIA, IN:PLAY_MUSIC, IN:PREFER, IN:PREVIOUS_TRACK_MUSIC, IN:QUESTION_MUSIC, IN:QUESTION_NEWS, IN:REMOVE_FROM_PLAYLIST_MUSIC, IN:REPEAT_ALL_MUSIC, IN:REPEAT_ALL_OFF_MUSIC, IN:REPLAY_MUSIC, IN:RESTART_TIMER, IN:RESUME_CALL, IN:RESUME_MUSIC, IN:RESUME_TIMER, IN:REWIND_MUSIC, IN:SEND_MESSAGE, IN:SET_AVAILABLE, IN:SET_DEFAULT_PROVIDER_CALLING, IN:SET_DEFAULT_PROVIDER_MUSIC, IN:SET_RSVP_INTERESTED, IN:SET_RSVP_NO, IN:SET_RSVP_YES, IN:SET_UNAVAILABLE, IN:SHARE_EVENT, IN:SILENCE_ALARM, IN:SKIP_TRACK_MUSIC, IN:SNOOZE_ALARM, IN:START_SHUFFLE_MUSIC, IN:STOP_MUSIC, IN:STOP_SHUFFLE_MUSIC, IN:SUBTRACT_TIME_TIMER, IN:SWITCH_CALL, IN:UNLOOP_MUSIC, IN:UPDATE_ALARM, IN:UPDATE_CALL, IN:UPDATE_METHOD_CALL, IN:UPDATE_REMINDER, IN:UPDATE_REMINDER_DATE_TIME, IN:UPDATE_REMINDER_LOCATION, IN:UPDATE_REMINDER_TODO, IN:UPDATE_TIMER
Canonical slot types: SL:AGE, SL:ALARM_NAME, SL:AMOUNT, SL:ATTENDEE, SL:ATTENDEE_EVENT, SL:ATTRIBUTE_EVENT, SL:CATEGORY_EVENT, SL:CONTACT, SL:CONTACT_ADDED, SL:CONTACT_METHOD, SL:CONTACT_RELATED, SL:CONTACT_REMOVED, SL:CONTENT_EXACT, SL:DATE_TIME, SL:DATE_TIME，41:68:SL:TODO, SL:EDUCATION_DEGREE, SL:EMPLOYER, SL:GENDER, SL:GROUP, SL:JOB, SL:LIFE_EVENT, SL:LOCATION, SL:MAJOR, SL:METHOD_RECIPES, SL:METHOD_RETRIEVAL_REMINDER, SL:METHOD_TIMER, SL:MUSIC_ALBUM_MODIFIER, SL:MUSIC_ALBUM_TITLE, SL:MUSIC_ARTIST_NAME, SL:MUSIC_GENRE, SL:MUSIC_PLAYLIST_MODIFIER, SL:MUSIC_PLAYLIST_TITLE, SL:MUSIC_PROVIDER_NAME, SL:MUSIC_RADIO_ID, SL:MUSIC_REWIND_TIME, SL:MUSIC_TRACK_TITLE, SL:MUSIC_TYPE, SL:NAME_APP, SL:NEWS_CATEGORY, SL:NEWS_REFERENCE, SL:NEWS_SOURCE, SL:NEWS_TOPIC, SL:NEWS_TYPE, SL:ORDINAL, SL:PERIOD, SL:PERSON_REMINDED, SL:PHONE_NUMBER, SL:RECIPES_ATTRIBUTE, SL:RECIPES_COOKING_METHOD, SL:RECIPES_CUISINE, SL:RECIPES_DIET, SL:RECIPES_DISH, SL:RECIPES_EXCLUDED_INGREDIENT, SL:RECIPES_INCLUDED_INGREDIENT, SL:RECIPES_MEAL, SL:RECIPES_QUALIFIER_NUTRITION, SL:RECIPES_RATING, SL:RECIPES_SOURCE, SL:RECIPES_TYPE, SL:RECIPES_TYPE_NUTRITION, SL:RECIPES_UNIT_MEASUREMENT, SL:RECIPES_UNIT_NUTRITION, SL:RECIPIENT, SL:SCHOOL, SL:SENDER, SL:SIMILARITY, SL:TIMER_NAME, SL:TITLE_EVENT, SL:TODO, SL:TYPE_CONTACT, SL:TYPE_CONTENT, SL:TYPE_RELATION, SL:USER_ATTENDEE_EVENT, SL:WEATHER_ATTRIBUTE, SL:WEATHER_TEMPERATURE_UNIT

Tokens:
{numbered_tokens}<|im_end|>
<|im_start|>assistant
<think>

</think>

JSON:
```

## 统计口径

顺序矩阵的原分数是本语主指标的百分数，差值是百分点。主要定点对照用 5,000 次同题配对重采样，区间为 95%。XNLI 按规范化前提组，TyDiQA 按文章标题，槽位和实体按同题重采样后重算聚合分母。每语后续回落的区间按该语可比较的后续阶段数校正。宏平均对语言等权。MASSIVE 宏平均的配对重采样使用共享源 ID。
