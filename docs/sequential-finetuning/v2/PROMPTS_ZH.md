# Benchmark V2 的完整训练输入示例

以下记录直接从冻结训练文件的词元序列解码。代码块保留分词器实际输入的换行、ChatML 角色和答题前缀。每个监督块包含答案及训练中真实出现的结束标记。用于测试时，把该语言的评测输入填入同一条指令，只向模型提供输入块。 [数据划分、生成参数与评分](PROTOCOL_ZH.md)

## MASSIVE 德语

训练题 `15391`。输入前缀为 598 个词元，监督答案为 18 个词元。

输入前缀

```text
<|im_start|>user
Identify the intent and extract the slots of the utterance. Return one JSON object with exactly two keys: "intent" and "slots". Use one canonical intent from the list. Each slot is [canonical slot type, exact text copied from the utterance, occurrence]. Occurrence starts at 1 and counts exact word-sequence matches in the utterance from left to right. Occurrence is counted separately for each exact text, never as the slot number. If two different texts each appear once, both slots have occurrence 1. Output the slots in utterance order. Use an empty list when no slot applies.
Example utterance: set an alarm for tomorrow at seven
Example JSON: {"intent":"alarm_set","slots":[["date","tomorrow",1],["time","seven",1]]}
Canonical intents: alarm_query, alarm_remove, alarm_set, audio_volume_down, audio_volume_mute, audio_volume_other, audio_volume_up, calendar_query, calendar_remove, calendar_set, cooking_query, cooking_recipe, datetime_convert, datetime_query, email_addcontact, email_query, email_querycontact, email_sendemail, general_greet, general_joke, general_quirky, iot_cleaning, iot_coffee, iot_hue_lightchange, iot_hue_lightdim, iot_hue_lightoff, iot_hue_lighton, iot_hue_lightup, iot_wemo_off, iot_wemo_on, lists_createoradd, lists_query, lists_remove, music_dislikeness, music_likeness, music_query, music_settings, news_query, play_audiobook, play_game, play_music, play_podcasts, play_radio, qa_currency, qa_definition, qa_factoid, qa_maths, qa_stock, recommendation_events, recommendation_locations, recommendation_movies, social_post, social_query, takeaway_order, takeaway_query, transport_query, transport_taxi, transport_ticket, transport_traffic, weather_query
Canonical slot types: alarm_type, app_name, artist_name, audiobook_author, audiobook_name, business_name, business_type, change_amount, coffee_type, color_type, cooking_type, currency_name, date, definition_word, device_type, drink_type, email_address, email_folder, event_name, food_type, game_name, game_type, general_frequency, house_place, ingredient, joke_type, list_name, meal_type, media_type, movie_name, movie_type, music_album, music_descriptor, music_genre, news_topic, order_type, person, personal_info, place_name, player_setting, playlist_name, podcast_descriptor, podcast_name, radio_name, relation, song_name, sport_type, time, time_zone, timeofday, transport_agency, transport_descriptor, transport_name, transport_type, weather_descriptor
Utterance: habe ich irgendwelche erwähnungen auf twitter<|im_end|>
<|im_start|>assistant
<think>

</think>

JSON:
```

监督文本

```text
 {"intent":"social_query","slots":[["media_type","twitter",1]]}<|im_end|>
```

## MasakhaNER 斯瓦希里语

训练题 `swa/train/146`。输入前缀为 205 个词元，监督答案为 17 个词元。

输入前缀

```text
<|im_start|>user
Identify every named entity in this numbered token sequence. Use PER for a person, ORG for an organization, LOC for a location, and DATE for a date or time expression. Return only a JSON list of [start,end,"TYPE"] triples. Token numbering starts at 0. The start is included and the end is excluded. Return [] if there is no entity.

Tokens:
0:Eric 1:Ries 2:muandishi 3:wa 4:kitabu 5:cha 6:"""" 7:The 8:Lean 9:Startup 10:"""" 11:anasema 12:teknolojia 13:mwaka 14:huu 15:imeokoa 16:maisha 17:ya 18:watu 19:wengi 20:sana 21:.<|im_end|>
<|im_start|>assistant
<think>

</think>

Entities:
```

监督文本

```text
 [[0,2,"PER"],[13,15,"DATE"]]<|im_end|>
```

## MasakhaNER 伊博语

训练题 `ibo/train/1401`。输入前缀为 385 个词元，监督答案为 2 个词元。

输入前缀

```text
<|im_start|>user
Identify every named entity in this numbered token sequence. Use PER for a person, ORG for an organization, LOC for a location, and DATE for a date or time expression. Return only a JSON list of [start,end,"TYPE"] triples. Token numbering starts at 0. The start is included and the end is excluded. Return [] if there is no entity.

Tokens:
0:Mmemme 1:ahụ 2:bụ 3:nke 4:e 5:mere 6:n 7:"""" 8:ịgbaso 9:ntuziaka 10:gọọmenti 11:nyere 12:banyere 13:ọrịa 14:korona 15:, 16:ma 17:bụrụkwa 18:nke 19:a 20:nọrọ 21:na 22:ya 23:kesàára 24:ndị 25:sonyere 26:na 27:ya 28:uwe 29:mkpuchi 30:ime 31:bụ 32:"""" 33:facemask 34:"""" 35:; 36:nwee 37:oge 38:ajụjụ 39:na 40:azịza 41:, 42:ma 43:gosikwa 44:ihe 45:nkiri 46:dị 47:iche 48:iche 49:banyere 50:ihe 51:ọgwụ 52:ike<|im_end|>
<|im_start|>assistant
<think>

</think>

Entities:
```

监督文本

```text
 []<|im_end|>
```

## MTOP 法语

训练题 `3231363734353432`。输入前缀为 1401 个词元，监督答案为 21 个词元。

输入前缀

```text
<|im_start|>user
Classify the utterance intent and extract the top-level slots. Return a JSON object with keys "intent" and "slots". Each slot is [start,end,"TYPE"], with zero-based token positions, inclusive start and exclusive end. Use the canonical labels exactly. Output slots in token order. Use [] when there is no slot.
Example tokens: 0:play 1:some 2:jazz
Example JSON: {"intent":"IN:PLAY_MUSIC","slots":[[2,3,"SL:MUSIC_GENRE"]]}
Canonical intents: IN:ADD_TIME_TIMER, IN:ADD_TO_PLAYLIST_MUSIC, IN:ANSWER_CALL, IN:CANCEL_CALL, IN:CANCEL_MESSAGE, IN:CREATE_ALARM, IN:CREATE_CALL, IN:CREATE_PLAYLIST_MUSIC, IN:CREATE_REMINDER, IN:CREATE_TIMER, IN:DELETE_ALARM, IN:DELETE_PLAYLIST_MUSIC, IN:DELETE_REMINDER, IN:DELETE_TIMER, IN:DISLIKE_MUSIC, IN:DISPREFER, IN:END_CALL, IN:FAST_FORWARD_MUSIC, IN:FOLLOW_MUSIC, IN:GET_AGE, IN:GET_AIRQUALITY, IN:GET_ALARM, IN:GET_ATTENDEE_EVENT, IN:GET_AVAILABILITY, IN:GET_CALL, IN:GET_CALL_CONTACT, IN:GET_CALL_TIME, IN:GET_CATEGORY_EVENT, IN:GET_CONTACT, IN:GET_CONTACT_METHOD, IN:GET_DATE_TIME_EVENT, IN:GET_DETAILS_NEWS, IN:GET_EDUCATION_DEGREE, IN:GET_EDUCATION_TIME, IN:GET_EMPLOYER, IN:GET_EMPLOYMENT_TIME, IN:GET_EVENT, IN:GET_GENDER, IN:GET_GROUP, IN:GET_INFO_CONTACT, IN:GET_INFO_RECIPES, IN:GET_JOB, IN:GET_LANGUAGE, IN:GET_LIFE_EVENT, IN:GET_LIFE_EVENT_TIME, IN:GET_LOCATION, IN:GET_LYRICS_MUSIC, IN:GET_MAJOR, IN:GET_MESSAGE, IN:GET_MESSAGE_CONTACT, IN:GET_MUTUAL_FRIENDS, IN:GET_RECIPES, IN:GET_REMINDER, IN:GET_REMINDER_AMOUNT, IN:GET_REMINDER_DATE_TIME, IN:GET_REMINDER_LOCATION, IN:GET_STORIES_NEWS, IN:GET_SUNRISE, IN:GET_SUNSET, IN:GET_TIMER, IN:GET_TRACK_INFO_MUSIC, IN:GET_UNDERGRAD, IN:GET_WEATHER, IN:HELP_REMINDER, IN:HOLD_CALL, IN:IGNORE_CALL, IN:IS_TRUE_RECIPES, IN:LIKE_MUSIC, IN:LOOP_MUSIC, IN:MERGE_CALL, IN:PAUSE_MUSIC, IN:PAUSE_TIMER, IN:PLAY_MEDIA, IN:PLAY_MUSIC, IN:PREFER, IN:PREVIOUS_TRACK_MUSIC, IN:QUESTION_MUSIC, IN:QUESTION_NEWS, IN:REMOVE_FROM_PLAYLIST_MUSIC, IN:REPEAT_ALL_MUSIC, IN:REPEAT_ALL_OFF_MUSIC, IN:REPLAY_MUSIC, IN:RESTART_TIMER, IN:RESUME_CALL, IN:RESUME_MUSIC, IN:RESUME_TIMER, IN:REWIND_MUSIC, IN:SEND_MESSAGE, IN:SET_AVAILABLE, IN:SET_DEFAULT_PROVIDER_CALLING, IN:SET_DEFAULT_PROVIDER_MUSIC, IN:SET_RSVP_INTERESTED, IN:SET_RSVP_NO, IN:SET_RSVP_YES, IN:SET_UNAVAILABLE, IN:SHARE_EVENT, IN:SILENCE_ALARM, IN:SKIP_TRACK_MUSIC, IN:SNOOZE_ALARM, IN:START_SHUFFLE_MUSIC, IN:STOP_MUSIC, IN:STOP_SHUFFLE_MUSIC, IN:SUBTRACT_TIME_TIMER, IN:SWITCH_CALL, IN:UNLOOP_MUSIC, IN:UPDATE_ALARM, IN:UPDATE_CALL, IN:UPDATE_METHOD_CALL, IN:UPDATE_REMINDER, IN:UPDATE_REMINDER_DATE_TIME, IN:UPDATE_REMINDER_LOCATION, IN:UPDATE_REMINDER_TODO, IN:UPDATE_TIMER
Canonical slot types: SL:AGE, SL:ALARM_NAME, SL:AMOUNT, SL:ATTENDEE, SL:ATTENDEE_EVENT, SL:ATTRIBUTE_EVENT, SL:CATEGORY_EVENT, SL:CONTACT, SL:CONTACT_ADDED, SL:CONTACT_METHOD, SL:CONTACT_RELATED, SL:CONTACT_REMOVED, SL:CONTENT_EXACT, SL:DATE_TIME, SL:DATE_TIME，41:68:SL:TODO, SL:EDUCATION_DEGREE, SL:EMPLOYER, SL:GENDER, SL:GROUP, SL:JOB, SL:LIFE_EVENT, SL:LOCATION, SL:MAJOR, SL:METHOD_RECIPES, SL:METHOD_RETRIEVAL_REMINDER, SL:METHOD_TIMER, SL:MUSIC_ALBUM_MODIFIER, SL:MUSIC_ALBUM_TITLE, SL:MUSIC_ARTIST_NAME, SL:MUSIC_GENRE, SL:MUSIC_PLAYLIST_MODIFIER, SL:MUSIC_PLAYLIST_TITLE, SL:MUSIC_PROVIDER_NAME, SL:MUSIC_RADIO_ID, SL:MUSIC_REWIND_TIME, SL:MUSIC_TRACK_TITLE, SL:MUSIC_TYPE, SL:NAME_APP, SL:NEWS_CATEGORY, SL:NEWS_REFERENCE, SL:NEWS_SOURCE, SL:NEWS_TOPIC, SL:NEWS_TYPE, SL:ORDINAL, SL:PERIOD, SL:PERSON_REMINDED, SL:PHONE_NUMBER, SL:RECIPES_ATTRIBUTE, SL:RECIPES_COOKING_METHOD, SL:RECIPES_CUISINE, SL:RECIPES_DIET, SL:RECIPES_DISH, SL:RECIPES_EXCLUDED_INGREDIENT, SL:RECIPES_INCLUDED_INGREDIENT, SL:RECIPES_MEAL, SL:RECIPES_QUALIFIER_NUTRITION, SL:RECIPES_RATING, SL:RECIPES_SOURCE, SL:RECIPES_TYPE, SL:RECIPES_TYPE_NUTRITION, SL:RECIPES_UNIT_MEASUREMENT, SL:RECIPES_UNIT_NUTRITION, SL:RECIPIENT, SL:SCHOOL, SL:SENDER, SL:SIMILARITY, SL:TIMER_NAME, SL:TITLE_EVENT, SL:TODO, SL:TYPE_CONTACT, SL:TYPE_CONTENT, SL:TYPE_RELATION, SL:USER_ATTENDEE_EVENT, SL:WEATHER_ATTRIBUTE, SL:WEATHER_TEMPERATURE_UNIT

Tokens:
0:Appelle 1:Ted 2:s' 3:il 4:te 5:plaît 6:.<|im_end|>
<|im_start|>assistant
<think>

</think>

JSON:
```

监督文本

```text
 {"intent":"IN:CREATE_CALL","slots":[[1,2,"SL:CONTACT"]]}<|im_end|>
```
