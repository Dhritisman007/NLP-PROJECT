# Annotation guidelines

You will see an **English source sentence** and its **machine translation** into German. Judge only whether the translation
**keeps the meaning of the source**. Do not judge style, fluency or word choice unless it changes the meaning.

For every item answer **"Meaning preserved?"**:

* **Yes**: everything important in the source is in the translation, nothing important is added or changed.
* **No**: the translation loses or changes meaning. Then tick every error type that applies (one or more):

| Type | Tick it when... | Example (source -> translation problem) |
|---|---|---|
| **Omission** | source content is missing from the translation | "She sold the car **and the house**" -> only the car is mentioned |
| **Mistranslation** | the meaning is changed: negation flipped, wrong word sense, who-did-what-to-whom swapped | "He did **not** attend" -> says he attended |
| **Entity / number** | a name, date, number, unit or currency is wrong or missing | "12 May 2022" -> a different date; "4.5 million" -> "45 million" |
| **Hallucination** | content appears in the translation that is not in the source | an extra clause or sentence that the source does not support |

Rules of thumb
* Minor grammar or awkward word order that does **not** change the meaning -> **Meaning preserved = Yes**.
* If you are unsure whether something is an error, choose the more serious reading only if a bilingual reader would be misled.
* Use "problem words" to copy the exact words that are wrong (optional but very helpful); use "notes" for anything unusual.
* Do not look up the reference translation or ask the other annotator: independent judgements are what make the agreement
  statistic meaningful. Some items are **attention checks** with a known answer; read every item carefully.
* Aim for about 2 minutes per item. You may stop and continue later (the HTML form saves your progress in the browser).

Return your file as `annotations_<your name>.csv` (HTML form: use the **Download CSV** button).
