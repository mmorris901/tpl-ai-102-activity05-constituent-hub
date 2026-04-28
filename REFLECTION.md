---
title: "Activity 5 Reflection - Constituent Services Hub"
type: reflection
version: "1.0.0"
---

# Activity 5 Reflection

Answer each question in 3-5 sentences. Thoughtful, specific responses earn full credit.

## 1. PII Redaction in Practice

What PII categories did the Azure AI Language service detect in the Memphis 311 complaints? Did you notice any false positives (non-PII flagged as sensitive) or false negatives (PII that was missed)? How might PII detection requirements differ between a government agency processing citizen complaints and a commercial customer service system?

The Azure AI Language service detected multiple PII categories including Person names (John Smith, Carlos Rivera, Nguyen Van Minh), US Social Security Numbers, physical Addresses (Main St, Beale Street, Oak Drive, Poplar Avenue), phone numbers, and PersonType entities (family, city crew). A few false positives appeared: "family" and "city crew" were flagged as PersonType entities when they're contextual references rather than identifiable individuals. The service successfully caught all primary PII (names, SSNs, addresses, phone numbers), showing minimal false negatives. Government agencies like Memphis 311 require strict PII redaction before any processing due to public records liability and privacy regulations, while commercial systems might only redact specific categories like SSNs, allowing customer names to flow through for support team context.

## 2. Sentiment as a Routing Signal

How could sentiment analysis be used to prioritize Memphis 311 complaints — for example, routing highly negative complaints to senior staff? What are the risks of relying solely on sentiment for prioritization (consider complaints that are neutral in tone but urgent in nature)? How would you combine sentiment with other signals for a more robust routing system?

Sentiment analysis could automatically escalate highly negative complaints to senior staff: the complaint about the broken streetlight using words like "completely unacceptable" and "dangerous" signals high urgency, while a neutral-positive complaint praising the water main repair crew can be logged without escalation. The risk is that urgent but neutral-toned complaints—such as "Pothole at Main St damaged my car"—might be under-prioritized because impact and actual damage matter more than emotional language. A robust routing system would combine sentiment with key phrases: complaints mentioning "fire," "flood," "emergency," or "safety" would automatically escalate regardless of sentiment, while key phrase extraction identifies domains (Public Works, Code Enforcement) for further routing.

## 3. Multilingual Challenges

How accurately did the Language service detect the language of short versus long text samples? What challenges might arise with code-switching (mixing languages in a single message), which is common in multilingual communities? How would you handle a complaint where language detection confidence is low?

The Language service accurately detected Spanish ("Hay un bache...") and Vietnamese ("Tôi muốn báo cáo...") with high confidence when given full sentences, but shorter phrases alone might produce lower confidence scores. Code-switching—common in bilingual Memphis communities saying things like "El bache on Poplar Avenue needs fixing"—presents a challenge because the service detects one primary language, potentially missing intent in the other language. For low-confidence language detection (below 0.7), the pipeline should either prompt human verification, attempt translation with fallback to original text if quality is low, or flag the complaint for manual review before routing.

## 4. CLU vs. Keyword Matching

Compare the CLU model's intent classification with the keyword-based fallback. In what scenarios did CLU perform better, and where did keyword matching suffice? What are the trade-offs of training and maintaining a custom CLU model versus using a simpler rule-based approach for a city's 311 system? *(If CLU was not configured in your environment, discuss how you would expect it to differ based on the training data in `data/intent_examples.json`.)*
If CLU were deployed, it would outperform keyword matching on nuanced complaints: detecting "I'm filing this to check on my complaint status" as check-status intent, whereas keyword matching might identify the word "filing" and misclassify as report-issue. Keyword matching succeeds for explicit complaints like "Report: pothole," but fails on requests like "Can you tell me the schedule for bulk pickup?" which are report-adjacent but actually ask-question intents. The trade-off is that CLU requires training data curation and model retraining when intent definitions change (costly for a city system), while keyword matching instantly adapts to new phrases but grows increasingly fragile and prone to errors as the complaint vocabulary expands.

## 5. Pipeline Design

Which step in your NLP pipeline was most critical to get right first, and why? If you were deploying this pipeline for production use handling thousands of complaints daily, what monitoring, fallback mechanisms, or error handling would you add? How would you measure pipeline health over time?

Step 1 (PII redaction) is most critical because it's a security gate—processing unredacted text through downstream services defeats the entire purpose and risks exposing SSNs and addresses in logs. For production, I would add: (1) retry logic with exponential backoff for API failures, (2) logging and alerting on high PII entity rates (potential spammy or test complaints), (3) sentiment + key phrase confidence thresholds that trigger manual review, (4) translation quality scoring to reject low-quality translations, and (5) CLU confidence thresholds that fall back to keyword matching. Pipeline health metrics would include: average processing time per complaint, entity detection rate trends, translation success rates, and intent classification agreement between CLU and keyword fallback (divergence signals model drift).