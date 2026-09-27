# Mandarin Learning Portal — Product Vision & Technical Specification

## 1. Vision

Extend the existing Mandarin Anki repository with a modern, intuitive web portal for learning **Mandarin Chinese (Simplified)** from English using flashcards and spaced repetition.

The repository already contains the canonical learning material and supports Anki. The new portal must build on this foundation rather than replace it.

The desired long-term workflow is:

```text
Attend physical Mandarin class
        ↓
Photograph / scan new learning material
        ↓
AI extracts, cleans and structures the material
        ↓
Review proposed changes
        ↓
Update the repository
        ↓
Generate required audio and other derived assets
        ↓
Automatically validate and deploy
        ↓
Learn using:
  ├── the web portal
  └── Anki
```

The overall goal is:

> Provide the most frictionless, effective, and intuitive personal environment possible for continuously learning Mandarin, while keeping the repository as the durable source of truth and preserving full Anki compatibility.

---

# 2. Existing Repository as Foundation

Before making architectural changes, inspect and understand the existing repository thoroughly.

The existing repository already provides important capabilities that must be preserved.

In particular:

```text
deck/notes.tsv
```

is the canonical source of learning-note data.

The existing repository also contains:

```text
anki/note-type/
media/
preview/
scripts/
```

and related tooling for:

- Anki import;
- Anki card templates;
- Anki styling;
- generated or manually supplied media;
- synchronizing repository media into Anki;
- card-template previews;
- image generation workflows;
- stable note identifiers.

The new portal must reuse these existing concepts wherever reasonable.

Do not create a second independent source of vocabulary or learning content.

---

# 3. Core Architectural Principle

The architecture should conceptually look like:

```text
                    ┌────────────────────┐
                    │   deck/notes.tsv   │
                    │  Source of Truth   │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
         Web Portal          Anki       Build Tooling
                              │
                              │
                       Existing workflow
```

The portal and Anki must consume the same canonical learning material.

Do not maintain:

```text
portal vocabulary
```

separately from:

```text
Anki vocabulary
```

unless a derived format is generated automatically from the canonical source.

---

# 4. Guiding Principles

## 4.1 Repository as Source of Truth

The Git repository remains authoritative for learning content.

This includes, where applicable:

- note IDs;
- Hanzi;
- Pinyin;
- English meanings;
- example sentences;
- example translations;
- audio references;
- images;
- tags;
- classifications;
- future metadata.

Changes made elsewhere must ultimately be reconciled back into the repository.

---

## 4.2 Preserve Existing Anki Compatibility

Anki compatibility is an existing core feature and must not regress.

The current workflow based on:

```text
deck/notes.tsv
```

must continue working.

Stable note IDs must continue to allow notes to be safely updated in Anki without losing scheduling and review history.

The portal should complement Anki rather than force migration away from it.

Users should be able to choose:

```text
Web Portal
```

or:

```text
Anki
```

for learning from the same content.

---

## 4.3 Static-First Architecture

The initial portal should preferably remain a **static Angular application deployed through GitHub Pages**.

Do not introduce a backend merely because a backend would be conventional.

First determine whether the requirements can be fulfilled with:

- static repository content;
- Angular;
- build-time transformations;
- browser-side persistence;
- generated static media;
- GitHub Actions;
- local or CI-based tooling.

A backend should only be introduced once a concrete requirement justifies it.

---

## 4.4 Learning UX Before Infrastructure

The primary purpose of the portal is learning Mandarin.

The learning experience should therefore receive more attention than infrastructure, dashboards or administration interfaces.

A normal study session should require almost no setup.

Advanced configuration should exist without dominating the interface.

---

## 4.5 Progressive Architecture

Design the system so that it can evolve approximately as follows:

```text
Personal static learning portal
        ↓
Personal multi-device portal
        ↓
Optional Capacitor mobile application
        ↓
Potential multi-user SaaS product
```

Do not prematurely implement features required only by the last stage.

---

# 5. Mandatory Research and Discovery Phase

**Do not start by implementing the portal UI.**

The first phase must investigate both the existing repository and appropriate UX/technical approaches.

Research and document:

1. the complete current repository structure;
2. the current `deck/notes.tsv` format;
3. existing scripts and npm commands;
4. existing Anki card templates;
5. the existing preview application;
6. current media conventions;
7. stable-ID conventions;
8. current image-generation workflows;
9. existing Anki synchronization workflows;
10. the sibling `@cynthion/ngx-formidable` repository;
11. modern flashcard/spaced-repetition UX;
12. current Anki learning concepts;
13. the current scheduling algorithm used by Anki;
14. mobile flashcard UX;
15. Mandarin-specific learning UX;
16. accessibility;
17. persistence for a static Angular SPA;
18. future cross-device synchronization options.

Create a short research and architecture document before substantial implementation begins.

Important decisions should record:

- problem;
- alternatives;
- advantages;
- disadvantages;
- chosen approach;
- rationale;
- future implications.

Do not blindly reproduce Anki's UI.

Reuse proven learning concepts while designing a cleaner and more focused experience.

---

# 6. Frontend Technology

Implement the portal as an **Angular SPA**.

Use the sibling library:

```text
@cynthion/ngx-formidable
```

where appropriate.

Before using or bypassing that library:

1. inspect its implementation;
2. understand its APIs;
3. understand its intended use cases;
4. determine where it fits naturally into the portal.

Do not create a competing form/configuration abstraction when the sibling library already solves the problem well.

---

# 7. Target Devices

The portal must work well on:

- desktop;
- tablet;
- mobile phone.

Responsive behavior is mandatory.

Mobile must be treated as a first-class learning environment rather than as a scaled-down desktop layout.

The study experience should work comfortably with:

- mouse;
- keyboard;
- touchscreen.

---

# 8. Primary Learning Experience

The core experience is an Anki-like flashcard study session.

Typical flow:

```text
Present prompt
      ↓
Learner actively recalls answer
      ↓
Learner reveals card
      ↓
Show answer and supporting information
      ↓
Learner rates recall
      ↓
Scheduling algorithm updates review state
      ↓
Present next card
```

The answer must not be shown before the learner explicitly reveals it.

Recall controls should only become available after reveal.

---

# 9. Existing Learning Data Model

The existing TSV data model must be treated as the starting point.

Current note fields include:

```text
id
hanzi
pinyin
meaning
example-hanzi
example-pinyin
example-meaning
audio
audio-example
image
tags
```

Do not replace this schema merely to make the Angular application easier to implement.

Instead determine whether:

1. the portal can consume this schema directly; or
2. a generated build-time JSON representation should be derived from it.

A likely architecture is:

```text
deck/notes.tsv
       ↓
Build-time parser / validator
       ↓
Generated portal data
       ↓
Angular application
```

Any generated portal-specific data must be treated as disposable build output.

The canonical representation remains `deck/notes.tsv`.

---

# 10. Stable IDs

The existing stable `id` field is critical.

IDs must never be regenerated simply because:

- wording changes;
- Pinyin changes;
- examples change;
- media changes;
- tags change;
- classifications change.

The portal should also use these stable IDs as the identity of learning items.

For example, browser review state should reference:

```text
noteId
```

rather than an array index or generated runtime identifier.

This enables future:

- synchronization;
- migration;
- Anki compatibility;
- content updates without losing learning progress.

---

# 11. Learning Material Classification

The existing `tags` column should be investigated before creating additional metadata fields.

Tags may already represent concepts such as:

```text
noun
verb
adjective
expression
grammar
lesson
topic
HSK level
```

Determine whether the existing tag system can support portal filtering cleanly.

Possible logical categories include:

- nouns;
- verbs;
- adjectives;
- adverbs;
- measure words;
- expressions;
- phrases;
- sentences;
- grammar constructs;
- other vocabulary.

Prefer leveraging existing tags over creating duplicate metadata.

If structured metadata beyond tags is genuinely needed, extend the canonical format deliberately and document the migration.

---

# 12. HSK Levels

The portal should support filtering by HSK level.

Determine whether HSK information should be represented using:

- tags;
- a new explicit TSV column;
- or another existing convention.

The choice should be made based on:

- maintainability;
- Anki compatibility;
- portal filtering;
- human readability;
- future ingestion automation.

HSK should remain metadata rather than the fundamental identity of a note.

Some classroom vocabulary may:

- have no official HSK classification;
- exceed the learner's current HSK level;
- not map neatly to a single HSK level.

Such notes must remain fully supported.

---

# 13. Learning Directions

The learner should be able to configure the study direction.

Important modes include:

```text
English → Mandarin
Mandarin → English
```

Within these modes, presentation settings may determine whether Mandarin is shown as:

```text
Hanzi
Pinyin
Hanzi + Pinyin
```

Potential combinations include:

```text
English → Hanzi
English → Pinyin
English → Hanzi + Pinyin

Hanzi → English
Pinyin → English
Hanzi + Pinyin → English
```

Do not necessarily expose every permutation as a separate permanent UI option.

Design the configuration so that it remains easy to understand.

---

# 14. Card Front

Depending on the selected learning mode, a card front may show:

- English meaning;
- Hanzi;
- Pinyin;
- Hanzi and Pinyin;
- image;
- other deliberately selected context.

The prompt should remain visually dominant.

Avoid accidentally revealing information that makes active recall trivial.

---

# 15. Card Back

After reveal, present useful supporting information.

Depending on available fields, this may include:

- Hanzi;
- Pinyin;
- English meaning;
- example sentence in Hanzi;
- example sentence in Pinyin;
- example sentence meaning;
- image;
- audio controls;
- relevant classifications or tags.

The primary answer must remain immediately identifiable.

Secondary metadata should not overwhelm the learning content.

---

# 16. Images

The repository already supports note images.

The portal should display images from the existing media workflow where present.

Do not create a second incompatible image-management system.

Existing image-generation tooling should be investigated and reused where appropriate.

The portal should gracefully support notes with:

```text
image present
```

and:

```text
no image
```

without visual awkwardness.

---

# 17. Recall Rating

After revealing a card, allow the learner to rate recall quality.

An Anki-like model should be considered:

```text
Again
Hard
Good
Easy
```

The exact UX should be informed by research.

Do not invent a scheduling algorithm casually.

Investigate the current Anki scheduling approach and determine whether an established implementation can reasonably be reused.

The scheduling implementation must be separated from UI components.

Conceptually:

```text
Study UI
   ↓
Learning Session Service
   ↓
Scheduling Abstraction
   ↓
Scheduling Algorithm
```

This allows the scheduling algorithm to evolve independently.

---

# 18. Study Queue

The portal should build a review queue according to:

- selected categories;
- selected HSK levels;
- learning direction;
- due dates;
- learning/review state;
- potentially new-card limits;
- configured scheduling behavior.

Do not simply shuffle the entire vocabulary list on every session.

The application should distinguish between:

- new cards;
- learning cards;
- review cards.

Exact terminology and behavior should follow the chosen scheduling algorithm.

---

# 19. Session Summary

At the end of a study session, optionally show a lightweight summary such as:

```text
Reviewed: 24
Again: 3
Hard: 4
Good: 14
Easy: 3
```

Avoid unnecessary gamification.

The purpose is to provide useful learning feedback, not to maximize engagement metrics.

---

# 20. Audio

Audio is an important part of learning Mandarin.

The existing repository already supports:

```text
audio
audio-example
```

and media files.

The portal should use these existing fields and files.

Audio should support:

1. pronunciation of the primary Mandarin content;
2. pronunciation of the example sentence.

Where configured, audio should play automatically.

---

# 21. Audio Settings

Settings should include at least:

- audio enabled/disabled;
- autoplay primary pronunciation;
- autoplay example sentence;
- manual replay;
- potentially playback speed.

Do not require runtime Azure calls from the browser during ordinary study.

Prefer static generated audio assets.

This provides:

- predictable performance;
- no browser-side credentials;
- GitHub Pages compatibility;
- easier Anki reuse;
- potential offline compatibility.

---

# 22. Existing Anki Audio Behavior

The existing Anki workflow includes audio-file support and a system-TTS fallback on supported platforms.

The new portal does not need to replicate Anki's implementation internally.

However, the canonical media fields should remain usable by both systems.

Where possible:

```text
same MP3
   ├── Anki
   └── Web Portal
```

rather than generating different files for each consumer.

---

# 23. Azure Text-to-Speech

Create or extend repository tooling for automatically generating Mandarin audio from canonical learning material.

Before implementing Azure integration, ask Chris which Azure resources are currently available.

Questions to resolve:

- Which Azure subscription is available?
- Is an Azure Speech resource already provisioned?
- Which region is it in?
- Which Mandarin voices are available?
- Is there an existing preferred voice?
- What cost constraints apply?
- Which credentials can safely be used locally?
- Which credentials can safely be stored as GitHub Actions secrets?

Then propose appropriate options before implementation.

---

# 24. Audio Generation Pipeline

The desired audio-generation workflow is approximately:

```text
deck/notes.tsv
       ↓
Find notes requiring audio
       ↓
Determine expected filenames
       ↓
Compare against media/audio/
       ↓
Generate only missing/outdated files
       ↓
Update TSV audio references if required
       ↓
Validate
```

The existing filename conventions should be investigated and preserved where practical.

Current workflows already use note-ID-based audio names.

Prefer deterministic naming based on stable IDs.

For example:

```text
<ID>.mp3
<ID>-ex.mp3
```

or whatever convention is already established in the repository.

---

# 25. Audio Regeneration

Do not regenerate every audio file on every build.

Audio tooling should determine whether audio needs to be generated because:

- no file exists;
- source Hanzi changed;
- source example changed;
- generation parameters changed;
- regeneration was explicitly requested.

The system should avoid unnecessary Azure usage.

---

# 26. Example Sentences

Each vocabulary item should support one high-quality example sentence initially, consistent with the current TSV schema.

If future requirements justify multiple examples, extend the data model deliberately rather than introducing portal-only examples.

The AI ingestion process should generate example content containing:

- Simplified Chinese;
- Hanyu Pinyin;
- English meaning.

The example should:

- contain the target vocabulary;
- sound natural;
- be useful in everyday Mandarin where possible;
- approximately fit the intended learner level;
- avoid unnecessarily advanced vocabulary;
- clearly demonstrate the target meaning.

---

# 27. Multiple Example Sentences

The long-term vision may require **1–2 examples per vocabulary item**.

However, the current repository schema appears to model a single example.

Do not introduce an incompatible representation immediately.

During the architecture phase, evaluate options such as:

### Option A — Keep one example for V1

Preserve the current model and add multiple examples only when needed.

### Option B — Extend TSV

Add explicit additional example columns.

### Option C — Introduce a richer canonical representation

Only if the TSV format genuinely becomes insufficient.

Do not choose Option C solely because JSON would be easier for the web application.

Preserving existing repository simplicity and Anki compatibility is important.

---

# 28. Material Ingestion Vision

A major goal is making post-class material ingestion extremely easy.

Target workflow:

```text
Physical Mandarin class
        ↓
Take photos / scan material
        ↓
Provide files to AI tooling
        ↓
Extract learning content
        ↓
Normalize Mandarin content
        ↓
Generate/validate Pinyin
        ↓
Generate English meaning
        ↓
Generate example sentence
        ↓
Categorize/tag
        ↓
Detect existing notes
        ↓
Generate proposed TSV changes
        ↓
Human review
        ↓
Apply changes
        ↓
Generate audio
        ↓
Validate repository
```

The human should spend most of the time reviewing rather than manually transcribing.

---

# 29. Human Review Requirement

AI output must not be inserted blindly into canonical learning material.

Prefer:

```text
Source material
      ↓
AI-generated proposal
      ↓
Diff / review
      ↓
Human approval
      ↓
deck/notes.tsv
```

AI-generated information should be particularly reviewable for:

- Hanzi;
- Pinyin;
- tones;
- translation;
- grammar;
- example sentences;
- tags;
- HSK classification;
- OCR interpretation.

---

# 30. Protect Manual Corrections

Once learning content has been manually corrected, AI tooling must not silently overwrite it.

The ingestion process should distinguish between:

```text
new content
```

and:

```text
existing canonical content
```

When an incoming item appears to match an existing note, show the proposed differences rather than replacing it automatically.

---

# 31. Duplicate Detection

The ingestion workflow should detect likely duplicates.

Potential comparison signals include:

- exact Hanzi match;
- normalized Hanzi match;
- identical meaning;
- identical Pinyin;
- known note IDs;
- aliases;
- highly similar phrases.

Do not automatically merge uncertain matches.

Present ambiguous cases for review.

---

# 32. Stable ID Generation

New learning items require stable IDs.

Investigate the current repository's ID convention and preserve it.

New IDs should:

- be unique;
- never be reused;
- not change when note contents change;
- work cleanly with existing Anki imports;
- be easy for tooling to generate.

Do not migrate existing IDs without a compelling reason.

---

# 33. AI Responsibilities During Ingestion

The AI-assisted ingestion tooling may:

- extract text from source images/documents;
- correct obvious OCR errors;
- identify vocabulary;
- identify phrases;
- identify expressions;
- identify example sentences;
- generate missing Pinyin;
- validate Pinyin;
- normalize Simplified Chinese;
- generate English meanings;
- generate example sentences;
- categorize content;
- generate tags;
- estimate HSK level;
- detect probable duplicates;
- prepare TSV changes;
- flag ambiguous source material.

The AI must explicitly surface uncertainty rather than inventing confident answers.

---

# 34. Paperless-ngx Evaluation

Investigate whether **paperless-ngx** adds meaningful value to the ingestion workflow.

Possible architecture:

```text
Phone
  ↓
paperless-ngx
  ↓
OCR / document processing
  ↓
API
  ↓
AI ingestion tooling
```

Alternative:

```text
Phone
  ↓
Image/PDF
  ↓
AI ingestion tooling
```

Do not introduce paperless-ngx unless it provides clear value.

Evaluate:

- upload convenience;
- OCR quality;
- document organization;
- preprocessing;
- API integration;
- operational complexity.

The desired system should remain as simple as practical.

---

# 35. Simplest Possible Classroom Workflow

Optimize toward:

```text
Take photos
   ↓
Provide photos
   ↓
Run one ingestion command/workflow
   ↓
Review proposed changes
   ↓
Approve
```

Avoid workflows requiring repeated manual conversion or copying between tools.

---

# 36. Portal Settings

The application needs an intuitive configuration experience.

Research whether settings belong in:

- a sidebar;
- a settings page;
- a pre-session configuration screen;
- contextual controls;
- some combination.

Settings should not clutter active studying.

---

# 37. Learning Settings

Settings should include at least:

- learning direction;
- Hanzi display;
- Pinyin display;
- learning categories/tags;
- HSK level;
- new-card behavior;
- review behavior.

Use sensible defaults.

A learner should be able to start studying immediately without first understanding every setting.

---

# 38. Scheduling Settings

Expose only scheduling controls that are genuinely useful.

Potential advanced configuration may include Anki-like concepts.

Keep advanced scheduling configuration separate from everyday settings.

Avoid forcing users to understand algorithm internals.

---

# 39. Audio Settings

Provide settings for:

- autoplay;
- autoplay primary content;
- autoplay example;
- manual replay;
- potentially playback speed.

Settings should persist between sessions.

---

# 40. Internationalization

The portal UI must support:

- English;
- Simplified Chinese.

This concerns the **application interface**, not merely learning material.

The interface language and learning direction are independent.

Examples:

```text
UI: English
Learning: English → Mandarin
```

and:

```text
UI: 简体中文
Learning: Mandarin → English
```

must both work.

Do not hardcode user-facing strings directly throughout components.

Use an appropriate Angular localization strategy.

---

# 41. Persistence

Learning progress must survive:

- page reloads;
- closing the browser;
- restarting the device.

The first version should preferably use browser-side persistence.

Likely technologies:

```text
IndexedDB
```

for learning state and:

```text
localStorage
```

for lightweight preferences if appropriate.

Research and document the final choice.

---

# 42. Content vs. Learning State

Maintain a strict conceptual separation.

## Canonical repository content

```text
deck/notes.tsv
media/
anki/
```

contains shared learning material.

## Browser learning state

Contains user/device-specific information such as:

- next review;
- review history;
- interval;
- ease/difficulty state;
- learning stage;
- selected settings.

Conceptually:

```text
Git Repository
├── notes
├── templates
├── media
└── shared metadata

Browser
├── scheduling state
├── review history
└── user preferences
```

---

# 43. Persistence Abstraction

Do not access IndexedDB directly from arbitrary UI components.

Provide an abstraction such as:

```text
LearningStateRepository
SettingsRepository
```

This will make it possible to later replace:

```text
local persistence
```

with:

```text
synchronized persistence
```

without redesigning the learning domain.

---

# 44. Content Version Changes

Repository learning content may change after a user has already studied it.

The application must handle situations such as:

```text
meaning updated
example corrected
Pinyin corrected
audio changed
tag added
```

without losing learning progress.

Stable note IDs should make this possible.

Determine how the portal detects newly added and deleted notes.

---

# 45. Multi-Device Synchronization

Cross-device portal synchronization is an open architectural question.

Do not automatically introduce a backend.

Potential progression:

## Phase 1

Device-local browser persistence.

```text
Desktop progress != Phone progress
```

unless manually transferred.

## Phase 2

Manual progress export/import.

Example:

```text
Export learning state
       ↓
JSON file
       ↓
Import on another device
```

## Phase 3

Automatic cloud synchronization.

Potential requirements:

- authentication;
- server-side persistence;
- conflict resolution;
- API;
- user accounts.

Do not implement Phase 3 until required.

---

# 46. Anki Synchronization Is Separate

Anki already has its own synchronization model through AnkiWeb.

Do not confuse:

```text
Portal learning-state synchronization
```

with:

```text
Anki synchronization
```

They are separate systems.

V1 does not need to synchronize review history between Anki and the portal unless explicitly requested.

They share **learning content**, not necessarily review state.

---

# 47. Future Review-State Interoperability

Longer term, investigate whether meaningful interoperability with Anki review history is possible or desirable.

Do not make this a V1 requirement.

Potential complexity includes:

- differing scheduling implementations;
- differing review histories;
- note vs. card state;
- synchronization conflicts.

Keep this separate from preserving Anki content compatibility.

---

# 48. Repository Parsing

Implement a well-tested parser for:

```text
deck/notes.tsv
```

Do not parse the TSV ad hoc inside Angular components.

A build-time transformation may generate something like:

```text
dist-data/notes.json
```

or another generated representation.

The generated file should not become canonical.

---

# 49. Content Validation

Provide automated validation for canonical repository content.

Validation should include at least:

- duplicate IDs;
- malformed TSV;
- missing required columns;
- invalid rows;
- missing Hanzi where expected;
- missing meaning;
- malformed tags;
- broken media references;
- referenced audio file missing;
- referenced image file missing;
- duplicate or suspicious content.

Where useful, classify messages as:

```text
ERROR
WARNING
INFO
```

---

# 50. Media Validation

Validate media references against repository files.

For example:

```text
audio field
   ↓
extract filename
   ↓
verify media/audio/<filename>
```

and:

```text
image field
   ↓
extract source
   ↓
verify media/images/<filename>
```

Ensure validation remains compatible with the existing Anki syntax.

---

# 51. Existing Media Sync

The current media synchronization functionality for copying repository files into Anki's `collection.media` must continue working.

Do not restructure media paths without accounting for:

- existing scripts;
- existing notes;
- Anki references;
- documentation.

Prefer portal support for the existing directory layout.

---

# 52. Existing Image Workflow

The repository already supports a sprite-based generative-image workflow.

Treat it as an existing capability.

Do not redesign image generation as part of the portal unless doing so materially improves the workflow.

At minimum ensure portal development does not break:

- sprite generation;
- sprite slicing;
- note image application;
- media synchronization.

---

# 53. Existing Anki Preview

The repository already includes browser-based card-template preview functionality.

Investigate whether any parts are reusable.

However:

```text
Anki template preview
```

and:

```text
Angular learning portal
```

serve different purposes.

Do not force them into one application merely to reduce the number of tools.

They may remain separate if that keeps both simpler.

---

# 54. Anki Card Templates

Existing Anki templates and CSS remain authoritative for Anki presentation.

The Angular portal may use its own visual components.

Do not attempt to render Angular components inside Anki.

Likewise, do not unnecessarily force the portal to use Anki's HTML templates.

Shared assets and design concepts may be reused where practical.

---

# 55. Portal UX

The visual design should be:

- clean;
- calm;
- focused;
- modern;
- fast;
- uncluttered;
- accessible;
- touch-friendly.

During active study, the card should dominate the screen.

Avoid dashboard-like information overload.

---

# 56. Desktop UX

Desktop should support efficient keyboard-driven learning.

Potential shortcuts:

```text
Space / Enter → reveal answer
1             → Again
2             → Hard
3             → Good
4             → Easy
R             → replay audio
```

Research and validate the final shortcuts.

Display shortcut hints subtly where useful.

---

# 57. Mobile UX

Touch interaction should be comfortable.

Important controls must:

- have sufficient touch area;
- be reachable;
- avoid accidental taps;
- work in portrait orientation;
- adapt sensibly to landscape orientation.

Do not make hover interactions essential.

---

# 58. Accessibility

Follow modern web accessibility practices.

At minimum consider:

- semantic HTML;
- keyboard navigation;
- visible focus;
- screen-reader support;
- sufficient contrast;
- touch target size;
- reduced-motion preference;
- controls not dependent solely on color;
- appropriate language metadata.

Mandarin content should use appropriate `lang` attributes where practical.

---

# 59. Typography

Chinese characters, Pinyin and English must remain visually distinguishable and readable.

Pay particular attention to:

- Hanzi font rendering;
- Pinyin tone marks;
- line height;
- character size;
- mobile readability;
- example-sentence layout.

Do not sacrifice Mandarin readability for decorative typography.

---

# 60. User Documentation

Help must be accessible from the portal.

Users should be able to understand:

- how studying works;
- how reveal works;
- what each rating means;
- how cards are scheduled;
- how filters work;
- what Hanzi/Pinyin options mean;
- how audio works;
- where learning progress is stored.

Use contextual help for small concepts and a dedicated help area for larger topics.

---

# 61. Repository Documentation

The repository itself should contain clear technical documentation.

Suggested documentation areas:

```text
docs/
├── architecture/
├── portal/
├── learning-material/
├── ingestion/
├── audio/
├── anki/
└── development/
```

Adapt to the current repository structure instead of restructuring unnecessarily.

---

# 62. README.md

The existing README already contains extensive Anki instructions.

Do not replace useful documentation with generic portal documentation.

Instead reorganize or extend it carefully if necessary.

The top-level README should eventually explain that the repository supports:

```text
Canonical Mandarin learning material
├── Anki
└── Web Portal
```

Detailed instructions may then link to focused documentation.

---

# 63. CLAUDE.md

Create or maintain a repository-level:

```text
CLAUDE.md
```

that explains the constraints Claude Code must respect.

It should include:

- repository purpose;
- source-of-truth rule;
- stable-ID rule;
- canonical TSV location;
- important scripts;
- media conventions;
- Anki compatibility requirements;
- portal architecture;
- test commands;
- validation commands;
- documentation expectations.

Keep it concise enough that it remains useful as operational context.

Do not duplicate the entire README.

---

# 64. GitHub Pages Deployment

Deploy the Angular portal through GitHub Pages.

The deployment should work automatically through GitHub Actions.

Ensure correct handling of Angular routing under a repository-specific GitHub Pages base path.

Prefer approaches that work reliably with static hosting.

---

# 65. CI Pipeline

The GitHub Actions pipeline should conceptually perform:

```text
Checkout
   ↓
Install dependencies
   ↓
Validate canonical learning material
   ↓
Lint
   ↓
Run tests
   ↓
Build portal
   ↓
Verify production output
   ↓
Deploy GitHub Pages
```

CI must fail when canonical learning data is structurally invalid.

---

# 66. Audio Generation and CI

Do not automatically generate every audio file during every portal build.

Audio generation should preferably be an explicit or content-triggered workflow.

Possible model:

```text
New notes merged
     ↓
Detect missing audio
     ↓
Generate audio
     ↓
Commit/update generated assets
```

or another controlled workflow.

The regular portal deployment should not need Azure credentials if all required static audio already exists.

---

# 67. Performance

The study flow should feel immediate.

Once a session starts:

- switching cards should not require server requests;
- revealing a card should be instant;
- audio should start promptly;
- the next card should be prepared where practical.

Consider preloading:

- next-card data;
- relevant audio;
- relevant image.

Avoid excessive up-front loading if the deck eventually becomes large.

---

# 68. Offline Capability

Investigate whether a PWA provides sufficient value.

A future target experience could be:

```text
Open portal while online
      ↓
Application + material cached
      ↓
Study offline
      ↓
Progress stored locally
```

Do not introduce complex offline synchronization prematurely.

Document whether basic PWA support is worthwhile after the learning MVP exists.

---

# 69. Angular Architecture

Keep UI, learning logic and persistence separated.

A conceptual architecture:

```text
┌──────────────────────────────────┐
│             Angular UI           │
├──────────────────────────────────┤
│        Application Services      │
│                                  │
│ Session / Filtering / Settings   │
├──────────────────────────────────┤
│             Domain               │
│                                  │
│ Note / Card / Review / Schedule  │
├──────────────────┬───────────────┤
│ Content          │ Learning      │
│ Repository       │ State Repo    │
├──────────────────┼───────────────┤
│ Static generated │ IndexedDB     │
│ data             │ initially     │
└──────────────────┴───────────────┘
```

Avoid putting scheduling logic into Angular components.

---

# 70. Suggested Angular Areas

The exact structure should follow current Angular best practices and the actual project.

Conceptually, features may include:

```text
study/
settings/
help/
shared/
```

Domain/application concepts may include:

```text
learning-content
study-session
scheduler
learning-state
audio
settings
```

Do not create abstractions merely to satisfy a theoretical architecture diagram.

---

# 71. Forms

Use `@cynthion/ngx-formidable` where it naturally supports:

- settings;
- configuration;
- filters;
- future content-management workflows.

Do not introduce it into simple controls where it adds unnecessary complexity.

---

# 72. Testing Strategy

Testing should focus on meaningful behavior.

## Domain/unit tests

Test:

- scheduling;
- card selection;
- filter behavior;
- study direction;
- queue creation;
- data parsing;
- stable-ID handling;
- persistence mapping.

## Content validation tests

Test:

- TSV structure;
- IDs;
- media references;
- metadata;
- duplicate detection.

## UI tests

Test the critical learning sequence:

```text
card appears
→ reveal
→ answer appears
→ rating controls activate
→ rating submitted
→ schedule updated
→ next card appears
```

## End-to-end tests

Test at least:

- representative desktop viewport;
- representative mobile viewport.

Do not optimize for test-count metrics.

---

# 73. Security

Even though V1 is personal and static:

- never expose Azure credentials to the browser;
- never commit secrets;
- use GitHub secrets for CI credentials;
- validate ingestion input;
- treat OCR output as untrusted;
- do not execute generated AI content;
- escape/render content safely in the portal.

---

# 74. Developer Tooling UX

Repository tooling should provide clear output.

Example:

```text
✓ Loaded 143 notes
✓ 143 stable IDs valid
✓ No duplicate IDs found
✓ 128 primary audio files found
⚠ 15 primary audio files missing
✓ 124 example audio files found
⚠ 19 example audio files missing
⚠ 8 notes have no HSK classification

Validation completed with 42 warnings and 0 errors.
```

Tooling should make changes visible and understandable.

Avoid opaque automation.

---

# 75. Future Capacitor Application

The portal may later be packaged as a mobile application using Capacitor.

Do not implement this in V1.

Avoid architecture choices that unnecessarily depend on desktop-browser-only features.

Abstract browser-specific functionality where doing so has clear value.

---

# 76. Future Multi-User SaaS

A future version may allow other learners to use the service for a monthly fee.

Potential future functionality:

- accounts;
- authentication;
- cloud persistence;
- synchronized review state;
- authorization;
- subscription management;
- Stripe;
- APIs;
- server infrastructure.

This is explicitly outside V1.

Do not implement:

- authentication;
- Stripe;
- subscriptions;
- multi-tenancy;
- account management;
- SaaS administration.

Architectural decisions should merely avoid making these capabilities unnecessarily difficult later.

---

# 77. Backend Decision

The initial architectural assumption should be:

> **No backend is required for V1 unless a concrete requirement proves otherwise.**

A likely V1 architecture is:

```text
Git Repository
      │
      ├── deck/notes.tsv
      ├── media/
      ├── Anki assets
      └── Angular source
               │
               ▼
          GitHub Actions
               │
               ▼
          GitHub Pages
```

with browser-local review state.

---

# 78. Backend Triggers

Reconsider introducing a backend only when requirements such as these become necessary:

- automatic cross-device review synchronization;
- multiple users;
- authentication;
- subscriptions;
- server-side secrets at runtime;
- centralized user state;
- collaborative content;
- remote ingestion initiated from the portal.

Do not implement infrastructure ahead of these requirements.

---

# 79. Open Question — Browser Persistence

Research the best V1 persistence mechanism.

Likely approach:

```text
IndexedDB
```

for study state.

Potentially:

```text
localStorage
```

for small settings.

Important requirements:

- migrations;
- stable note-ID references;
- resilient updates;
- exportability;
- future synchronization compatibility.

---

# 80. Open Question — Cross-Device Sync

Determine with Chris whether automatic portal-progress synchronization is necessary for V1.

If not, prefer local persistence.

If a lightweight solution is desirable, investigate progress export/import before building a backend.

---

# 81. Open Question — Spaced Repetition

Research:

- the scheduling algorithm currently used by Anki;
- available mature implementations;
- licensing implications;
- required stored state;
- algorithm versioning;
- migration behavior;
- differences between note and card state.

Prefer an established algorithm rather than creating a custom one.

---

# 82. Open Question — Portal Cards vs. Anki Cards

The existing Anki setup can create multiple cards from one note.

The portal must decide whether review state belongs to:

```text
note
```

or:

```text
learning direction/card
```

For example:

```text
Mandarin → English
```

and:

```text
English → Mandarin
```

may require independent scheduling.

Research this explicitly.

Do not assume one scheduling record per TSV row is sufficient.

---

# 83. Open Question — Tags vs. Structured Metadata

Investigate how existing tags are currently used.

Determine whether concepts such as:

- part of speech;
- lesson;
- topic;
- HSK level;
- source class;
- grammar category

can remain tags.

Only add explicit TSV columns where structured semantics justify them.

---

# 84. Open Question — Multiple Examples

Determine whether one example per note remains sufficient.

Do not redesign the canonical format until a real need for multiple examples has been demonstrated.

If multiple examples are added, preserve:

- Git friendliness;
- Anki compatibility;
- simple manual editing;
- deterministic parsing.

---

# 85. Open Question — Audio Storage

The repository currently stores generated media.

Evaluate whether continuing this approach is appropriate as audio volume grows.

Possible future options:

- normal Git;
- Git LFS;
- GitHub release/build artifacts;
- object storage.

For V1, prefer existing repository conventions unless scale creates a real problem.

---

# 86. Open Question — OCR Pipeline

Compare:

```text
Photo → AI directly
```

against:

```text
Photo → paperless-ngx → OCR → AI
```

Prefer the simpler workflow unless paperless-ngx substantially improves accuracy or usability.

---

# 87. Suggested Repository Evolution

Do not restructure immediately.

After repository discovery, a possible eventual structure might resemble:

```text
/
├── README.md
├── CLAUDE.md
│
├── deck/
│   └── notes.tsv
│
├── anki/
│   └── note-type/
│
├── media/
│   ├── audio/
│   ├── images/
│   └── sprites/
│
├── preview/
│
├── portal/
│   └── Angular application
│
├── scripts/
│   ├── existing scripts
│   ├── validation/
│   ├── ingestion/
│   └── audio/
│
├── docs/
│   ├── architecture/
│   ├── portal/
│   ├── ingestion/
│   └── learning-material/
│
└── .github/
    └── workflows/
```

However:

> Adapt to the current repository. Do not perform a large restructuring merely to match this example.

---

# 88. Milestone 0 — Repository Discovery

Before major modifications:

- inspect all relevant repository directories;
- inspect `deck/notes.tsv`;
- inspect package scripts;
- inspect Anki templates;
- inspect media tooling;
- inspect sprite tooling;
- inspect preview tooling;
- inspect stable-ID conventions;
- inspect README documentation;
- inspect existing tests;
- inspect the sibling `@cynthion/ngx-formidable` repository.

Produce a concise repository assessment.

---

# 89. Milestone 1 — Research and Architecture

Research and document:

- study UX;
- responsive UX;
- scheduling algorithm;
- portal card identity;
- persistence;
- data parsing strategy;
- tag/metadata handling;
- GitHub Pages architecture;
- ingestion approach;
- Azure audio approach;
- multiple-example implications.

Produce architecture decisions before substantial implementation.

---

# 90. Milestone 2 — Portal Learning MVP

Implement a genuinely usable study experience.

Include:

- Angular application shell;
- responsive layout;
- canonical learning-data loading;
- card prompt;
- reveal;
- Mandarin → English;
- English → Mandarin;
- Hanzi/Pinyin display options;
- recall rating;
- spaced repetition;
- browser persistence;
- audio playback;
- image display;
- essential filters;
- essential settings.

At this milestone, the portal should already be useful for daily studying.

---

# 91. Milestone 3 — Internationalization and UX Polish

Implement:

- English UI;
- Simplified Chinese UI;
- responsive refinements;
- desktop keyboard controls;
- mobile touch refinements;
- accessibility improvements;
- contextual help;
- session summary.

---

# 92. Milestone 4 — Content Validation

Implement automated validation for:

- TSV format;
- IDs;
- duplicates;
- required values;
- tags/metadata;
- audio references;
- images;
- generated portal data.

Integrate validation into local development and CI.

---

# 93. Milestone 5 — AI Ingestion

Implement a simple AI-assisted workflow for classroom material.

Target:

```text
Photos/PDF
    ↓
AI extraction
    ↓
Structured proposal
    ↓
Duplicate detection
    ↓
TSV diff
    ↓
Human approval
```

Do not start with a large document-management system.

Solve the basic photo-to-reviewed-TSV workflow first.

---

# 94. Milestone 6 — Azure Audio Automation

After Chris confirms available Azure resources:

- evaluate available Mandarin voices;
- choose generation approach;
- implement audio generation;
- reuse stable note IDs;
- generate primary audio;
- generate example audio;
- skip existing valid audio;
- update references where necessary;
- document configuration.

Ensure generated audio works in both the portal and Anki.

---

# 95. Milestone 7 — CI/CD

Implement GitHub Actions for:

- install;
- validation;
- linting;
- tests;
- Angular production build;
- GitHub Pages deployment.

Keep Azure audio generation separate from ordinary deployment unless there is a compelling reason otherwise.

---

# 96. Milestone 8 — Final V1 Polish

Perform:

- mobile usability review;
- tablet review;
- desktop review;
- accessibility review;
- performance review;
- documentation review;
- Anki regression test;
- media workflow regression test;
- ingestion usability review.

Remove unnecessary complexity discovered during implementation.

---

# 97. Definition of Done for V1

V1 is successful when Chris can:

1. open the portal on desktop, tablet or mobile;
2. study the same learning material stored in `deck/notes.tsv`;
3. study Mandarin → English;
4. study English → Mandarin;
5. choose suitable Hanzi/Pinyin presentation;
6. filter the material meaningfully;
7. hear Mandarin pronunciation;
8. view available card images;
9. reveal answers before rating;
10. rate recall;
11. have future reviews scheduled appropriately;
12. close the browser and retain portal learning progress;
13. continue importing the same canonical material into Anki;
14. retain existing Anki scheduling when learning content is updated;
15. continue using repository media with Anki;
16. add new classroom material through a documented low-friction workflow;
17. generate missing audio through repository tooling;
18. push changes and automatically deploy the portal through GitHub Pages.

---

# 98. Non-Goals for V1

Do not implement unless explicitly requested:

- authentication;
- user accounts;
- Stripe;
- subscriptions;
- multi-tenancy;
- social features;
- leaderboards;
- elaborate gamification;
- native iOS application;
- native Android application;
- automatic portal/Anki review-history synchronization;
- complex cloud infrastructure;
- a backend without demonstrated need.

V1 should optimize for:

> **One learner learning Mandarin exceptionally well.**

---

# 99. Engineering Principles for Claude Code

When working on this repository:

1. Inspect before changing.
2. Understand existing workflows before replacing them.
3. Preserve `deck/notes.tsv` as the canonical learning-data source.
4. Preserve stable note IDs.
5. Preserve Anki compatibility.
6. Preserve working media workflows.
7. Avoid duplicate representations of canonical content.
8. Treat portal-specific JSON or indexes as generated data.
9. Keep learning content separate from user learning state.
10. Prefer established scheduling algorithms.
11. Prefer static architecture until a backend is justified.
12. Prefer simple solutions over speculative infrastructure.
13. Protect manually corrected learning content.
14. Make AI-generated content reviewable.
15. Never commit secrets.
16. Write tests for meaningful behavior.
17. Keep documentation synchronized with implementation.
18. Treat mobile as a first-class experience.
19. Do not prematurely build the future SaaS product.
20. Do not perform large repository restructurings without a concrete benefit.

---

# 100. Decisions Requiring Chris's Input

Do not block unrelated work on these questions.

## Azure

Before implementing Azure audio generation, ask Chris:

- Which Azure subscription/resource is available?
- Is Azure Speech already provisioned?
- Which region?
- Which voices/models are available?
- Is there a preferred Mandarin voice?
- What are acceptable costs?

Then propose suitable alternatives before implementing the integration.

---

## Portal Synchronization

Before adding server infrastructure, ask:

> Is automatic learning-progress synchronization between desktop, tablet and mobile required for V1, or is local per-device progress acceptable initially?

---

## Ingestion

After researching the simplest ingestion architecture, present the proposed workflow before adding infrastructure such as paperless-ngx.

Target:

```text
Take photos
   ↓
Run/import
   ↓
Review proposed learning material
   ↓
Approve
```

---

# 101. First Task for Claude Code

Start with **repository discovery and research**, not major implementation.

Perform the following:

1. inspect the complete current repository;
2. summarize its current architecture;
3. inspect `deck/notes.tsv` and document its actual schema;
4. determine the current stable-ID convention;
5. document exactly how Anki import and re-import currently work;
6. inspect existing Anki templates and CSS;
7. inspect the existing card-preview system;
8. inspect all existing npm scripts;
9. inspect media synchronization tooling;
10. inspect existing audio workflows;
11. inspect existing sprite/image tooling;
12. inspect the sibling `@cynthion/ngx-formidable` repository;
13. determine how the Angular portal can consume the existing canonical TSV without creating a second source of truth;
14. research an appropriate spaced-repetition implementation;
15. determine whether scheduling state should exist per note or per generated learning direction/card;
16. evaluate browser persistence;
17. evaluate future synchronization options without implementing them;
18. research responsive flashcard-learning UX;
19. research Mandarin-specific study UX;
20. evaluate the simplest photo/PDF ingestion workflow;
21. identify what existing repository functionality must remain untouched;
22. identify technical debt relevant to the portal;
23. propose the Angular architecture;
24. propose a phased implementation plan;
25. identify decisions that require Chris's input.

Do not perform a major repository restructuring during this first task.

Do not replace existing working Anki functionality simply because another design would be cleaner in isolation.

Present the findings, architecture proposal and important decisions before beginning consequential implementation.
