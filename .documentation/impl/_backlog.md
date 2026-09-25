<!-- markdownlint-disable link-image-style -->

# Backlog

> `bug` · `refactor` · `research` — items without a tag are implicitly `feature`

## Bugs

## Vision: Portal to Learn Flashcards English -> Chinese (Simplified)

- How to implement this portal:
  - configure Headroom globally, see <https://www.headroomlabs.ai/>
  - setup Docker Sandbox for safe, unattended AI execution, see <https://www.docker.com/products/docker-sandboxes/>
  - to interact with the repository in the sandbox, use VS Code SSH or similar
- Angular SPA application, using ngx-formidable
- self-learning portal with flashcards, Anki-like
  - user is presented the card and must recall the answer before flipping it
  - user can mark the card as known, unknown, etc. after flipping it; depending how well it went, it will come for repetition sooner
  - audio files run automatically
    - the word/sentence to learn
    - an example sentence containing the word/sentence to learn
- options (for configuring portal settings), e.g. on page or in sidebar, depending on what makes most sense
  - learning category (e.g. word types, expressions, )
  - pinyin vs hanzi
  - hsk level
  - Anki-like settings for learning algorithm
  - audio settings (autoplay, etc.)
- Goal: The portal supports me perfectly and intuitively when learning Chinese/Mandarin
- Sourcing:
  - after physical class, Chris takes a picture of the physical learning material and uses AI to bring it in shape
  - I want to have an as simple as possible solution for me to provide the new material I learned every class and that is automatically processed and fed into the portal; is a backend required?
  - optional, only if needed: photographed/scanned documents (with mobile phone) could go to paperless ngx (<https://docs.paperless-ngx.com/>) first or after, maybe its AI is already doing a lot of OCR done right, so it's a question of API
  - use a script to use Azure Subscription to translate the words/sentences into audio; ask me about what Azure model I actually have access to and propose possibilities for me how to set it up
  - AI will order, cleanup, place and categorize everything
  - this repository is the truth and source of all learning materials
  - a GitHub Action will build and deploy the portal (on GitHub pages)
- Currently, this repository is Anki-compatible and describes, how to use it. The project and portal must still allow for that and might even advertise/support it.
- There is i18n on the page that allows switching between English and Chinese (Simplified)
- The UX is super clean and easy to use. All controls define/describe clearly what they are and/or do.
- The UX and page structure must first be researched before any implementation starts.
- The portal must be usable on desktop, tablet and mobile. A responsive design is thus a must.
- The user documentation is easily accessible.
- There is technical documentation available in this repository. Especially stating how the learning material is sour
- At some point in the future, I might want to create a mobile app hosting the portal, using Capacitor.
- At some point in the future, I might want to let user's use the portal for a monthly fee. This will then have to be implemented, probably with an authentication and/or API key. This will then need to be done with Stripe.
- Open Questions:
  - persistence accross sessions
  - synchronization between devices -> web page
