# Document splitting rules

This document defines how a compiled Supreme Court filing PDF is divided into
logical documents. It describes the current canonical categories in
`configs/config.json` together with the deterministic reconciliation performed
by `structure_split.py` and `split_repair.py`.

The rules are about document identity and page ownership. They are not defect
rules and do not decide whether a filing is legally compliant.

## 1. Processing model

The production flow is:

1. Extract text, layout, headings, printed folios and image/OCR signals for each
   physical PDF page.
2. Propose contiguous document boundaries.
3. Classify each proposed segment using the controlled document taxonomy.
4. Reconcile the result against explicit headings, master-Index ranges, printed
   folios, document continuity and Annexure nesting.
5. Mark unsupported or conflicting assignments for review rather than inventing
   a document.
6. Slice the original PDF using the reconciled physical-page ranges.

The model output is a warm-start hint. It is not the final authority.

## 2. Global rules

### 2.1 Evidence priority

Use the following evidence order when signals conflict:

1. An already established enclosing Annexure or reproduced-record boundary.
2. A reliable master-Index row matched to printed folios and document content.
3. An explicit heading at an independently established document opening.
4. Matching court, case number, date, party context or issuing authority.
5. Consecutive printed pagination and textual/layout continuity.
6. Expected outer paper-book position.
7. Model classification alone.

No single keyword, case caption, date, signature or page position is sufficient
when the surrounding evidence contradicts it.

### 2.2 Start rule

A document starts only when the page is an actual independent opening. A title
mentioned in an Index row, chronology, prayer, quotation or narrative does not
create a boundary.

### 2.3 Continuation rule

After a document is established, keep its:

- captionless body pages;
- wrapped tables and rows;
- signature, verification, attestation and certification pages;
- OCR-poor pages supported by surrounding continuity;
- internal headings, quotations and references;
- pages with matching printed folios, court identity or source identity.

### 2.4 End rule

A document ends immediately before the next independently established outer
document. Do not end it merely because a continuation page repeats words such
as `ORDER`, `AFFIDAVIT`, `INDEX`, `ANNEXURE`, `PRAYER` or `CERTIFICATE`.

### 2.5 Current filing versus reproduced material

A document belonging to an earlier High Court, tribunal, district-court or
historical Supreme Court record normally stays inside the enclosing Annexure.
Its internal petition, affidavit, index, application, order, appearance memo or
vakalatnama must not become a new outer filing document.

### 2.6 Master Index

The master paper-book Index is used as structural evidence, not as proof that a
listed document actually exists. A range is applied only when its printed
folios, physical position and document content are sufficiently consistent.
An explicit conflicting outer heading prevents a weak or OCR-misaligned Index
range from overwriting that page.

### 2.7 Annexure precedence

Once an Annexure range is established, headings reproduced inside it remain
nested. A new outer boundary requires corroborating evidence such as a new
outer Annexure opening, matching master-Index range, source transition or an
authoritative document start.

The exception is the complete decision directly challenged by the current Main
Petition. When court, case, date, decision type, prayer and outer Index evidence
identify it, its primary category is `Impugned Order`, even if an Annexure
identifier is printed on its separator or cover.

### 2.8 Missing and ambiguous pages

Blank or OCR-damaged pages inherit a document only when surrounding evidence
establishes continuity. Missing OCR by itself is never proof of a document
type. If ownership cannot be safely determined, use `Unidentified`/review.

### 2.9 Repeated documents

Documents that can legitimately repeat—Applications, Annexures, separate
volume Indexes, separate appearance documents and optional later pleadings—must
remain separate when each has its own supported opening. Continuation pages do
not create a repeat.

### 2.10 Same-page documents

The internal page map can represent more than one label on a physical page when
two genuine documents share that page. This is exceptional. A dedicated rule
must resolve final ownership; for example, when Memo of Appearance and
Vakalatnama share one physical page, the page belongs to Vakalatnama.

## 3. Document-specific rules

### 3.1 Advocate's Checklist

**Identity:** The independent advocate/AOR compliance checklist for the current
Supreme Court filing.

**Start:** `CHECK LIST`, `CHECKLIST`, `CHECK-LIST` or a clear OCR variant,
supported by numbered compliance questions and `YES`, `NO` or `N.A.` answers
about limitation, fees, copies, certification or filing requirements.

**Continue:** All questions, continuation tables, the checklist's declaration,
date, AOR signature and advocate details.

**End:** Before the next independently established outer document. A missing
signature does not end or disqualify it.

**Exclude:** A Registry, government or lower-court checklist reproduced in
another record. A closing declaration inside the checklist is not AOR's
Declaration or AOR's Certificate.

**Repeat:** A separately presented checklist starts a new document.

### 3.2 Cover Page

**Identity:** The outer title sheet for the current paper book or an independent
volume.

**Start:** A cover-style combination of Supreme Court, jurisdiction,
case/parties and filing-advocate details, corroborated by `PAPER BOOK`,
`COVER PAGE`, `FOR INDEX PLEASE SEE INSIDE` or a verified volume opening.

**Continue:** Only an integral cover continuation.

**End:** Before the first contents table, form, pleading or other independent
document.

**Exclude:** A Supreme Court caption alone. Dedicated opening pages of the Main
Petition, Application or Affidavit belong to those documents. A reproduced
lower-court cover remains inside its source record.

**Repeat:** Every independently established volume may have its own Cover Page.

### 3.3 Record of Proceedings

**Identity:** A current-case Supreme Court hearing/proceedings record or the
paper book's outer Record of Proceedings register.

**Start:** `RECORD OF PROCEEDING`/`RECORD OF PROCEEDINGS`, supported by hearing
dates, appearances, directions or orders. A blank register with serial/date/page
columns can qualify when its outer position is independently established.

**Continue:** Appearances, order text, directions, signatures and captionless
continuation pages. There is no fixed page count.

**End:** Before a separately starting hearing record or another outer document.

**Exclude:** Counsel lists or hearing language inside a judgment; a petitioner
chronology; historical Supreme Court records reproduced inside an Annexure.
`ORDER` inside an established Record of Proceedings does not create an
Impugned Order.

**Repeat:** Each independently presented current-case hearing record is a
separate document.

### 3.4 AOR's Declaration

**Identity:** An independent advocate/AOR declaration concerning curing or
refiling defects in the current filing.

**Start:** `DECLARATION`, `DECLEARTION`, `DECLERATION` or a clear variant,
supported by language that defects were cured, changes were confined to curing
defects, or paper books are complete.

**Continue:** The declaration body, date, signature and contact details.

**End:** Before the next independently established outer document.

**Exclude:** A declaration inside Main Petition, Checklist or Listing Proforma;
a sworn Affidavit; or a pleadings-confined AOR's Certificate.

**Repeat:** A separately presented declaration starts another document.

### 3.5 AOR's Certificate

**Identity:** An independent Advocate-on-Record filing certificate for the
current proceeding.

**Start:** `CERTIFICATE` or a letter-spaced/OCR variant, together with language
certifying that the primary pleading is confined to pleadings/documents before
the challenged court or equivalent filing-certification language.

**Continue:** Complete certification, filing date, signature and advocate
details.

**End:** Before the next independently established outer document.

**Exclude:** A signature block alone; certificates internal to Checklist,
Listing Proforma, Annexure, Appendix or another document; defect-curing
declarations.

**Repeat:** A separately presented certificate starts another document.

### 3.6 Index

**Identity:** The independent contents index for the current paper book or
volume.

**Start:** A structured documents/particulars table with printed page or folio
ranges, normally headed `INDEX` or a clear variant.

**Continue:** The complete table, unheaded continuation pages, wrapped rows and
associated entries.

**End:** Before an actual independently established document or a separate
volume Index.

**Exclude:** An item merely listed in an Index row; internal indexes in
reproduced records. Determine function for `FILING INDEX`: a contents table is
Index, while a submission memo with filing advocate/date is Filing Memo.

**Repeat:** A distinct master/volume Index starts another document.

### 3.7 Office Report on Limitation

**Identity:** The independent Supreme Court Registry limitation report for the
current filing.

**Start:** `OFFICE REPORT ON LIMITATION`, `(O/R) ON LIMITATION` or a clear OCR
variant, supported by filing/refiling timeliness or delay fields.

**Continue:** All report fields, continuation pages and Registry certification.

**End:** Before the next independently established outer document.

**Exclude:** Ordinary limitation discussion, delay-condonation Applications,
`REPORT OF FRESH CASE`, `OFFICE REPORT OF FRESH CASE`, `FRESH CASE REPORT`, and
reports reproduced inside an exhibit.

**Repeat:** Only another independently established limitation report qualifies.
Never manufacture a missing report.

### 3.8 Listing Proforma

**Identity:** The independent first-listing information form for the current
filing.

**Start:** `LISTING PROFORMA/PERFORMA`, `PROFORMA/PERFORMA FOR FIRST LISTING`,
`PROFORMA OF FIRST LISTING`, `LISTED PROFORMA` or a clear variant, supported by
prompt/value fields about Acts, challenged court/order/date, judges, matter
nature, parties or connected cases.

**Continue:** All form fields, continuation pages, closing date, AOR code,
advocate details, declaration/certificate and signature.

**End:** Before the next independently established outer document.

**Exclude:** Bare `PROFORMA` without listing context. Its closing advocate
certificate/details remain Listing Proforma, not AOR's Certificate. A similar
form inside a reproduced record remains enclosed.

**Repeat:** A separately presented first-listing form starts another document.

### 3.9 Synopsis

**Identity:** The current filing's narrative summary of the dispute, issues and
reasons for relief.

**Start:** `SYNOPSIS` or a clear OCR variant followed by narrative summary
content. Under `SYNOPSIS AND LIST OF DATES`, narrative pages are Synopsis.

**Continue:** Narrative continuations, quoted decisions and document
references.

**End:** Before a separately introduced List of Dates chronology or another
outer document.

**Exclude:** Dates, Annexure references and quoted headings embedded in the
narrative. A Synopsis inside a reproduced lower-court record remains enclosed.

**Repeat:** Only an independently established outer Synopsis starts another
document.

### 3.10 List of Dates & Events

**Identity:** The current filing's chronological dates-and-events section.

**Start:** `LIST OF DATES`, `LIST OF DATES AND EVENTS`, `LIST OF DATES & EVENTS`
or `LIST OF EVENTS`, supported by dates/date ranges linked to events. A combined
Synopsis/List heading is assigned here when the page is chronological.

**Continue:** Complete rows, wrapped descriptions, quotations and unheaded
continuations. Every page need not repeat a date.

**End:** Before the next independently established outer document.

**Exclude:** Annexure references, quoted judgments and statements such as “the
impugned order was passed” do not start Annexure or Impugned Order. A chronology
inside a reproduced record remains enclosed.

**Repeat:** Only a separately established outer chronology starts another
document.

### 3.11 Impugned Order

`Impugned Order`, `Impugned Judgment` and `Impugned Judgement` are normalized to
the canonical label `Impugned Order`.

**Identity:** The complete decision directly challenged by the current Main
Petition.

**Preferred position:** After Synopsis/List of Dates and before Main Petition.
This is supporting evidence, not an absolute requirement.

**Start by Index:** A unique reliable master-Index row for the impugned
order/judgment, matched to its numeric printed-folio range and physical pages.

**Start by explicit heading:** `IMPUGNED ORDER`, `IMPUGNED JUDGMENT`,
`IMPUGNED JUDGEMENT`, their `FINAL` variants, or `CERTIFIED COPY OF THE
IMPUGNED ...`, when printed as the actual document opening.

**Start by court-record evidence:** A lower-court/tribunal/authority caption at
court folio 1, supported by `JUDGMENT`, `ORDER`, `ORDER SHEET`, `CORAM`,
reserved/pronounced/decision date, case number or neutral citation. A title on
folio 2 or 3 is accepted when intervening folios and neutral citation remain
consistent.

**Start by bounded position:** A lower-court decision between established
Synopsis/List of Dates and the current Main Petition can qualify when its court
folio starts at 1 and it has judgment/order evidence. If the preface is absent,
a strong lower-court judgment before the Main Petition may still qualify.

**Continue:** Party and appearance pages, judgment/order body, consecutive
court folios, matching `Page X of Y`, matching neutral citation, signatures and
integral certification. Up to three temporarily unreadable folio pages may be
bridged when later pagination proves continuity.

**End:** Before Main Petition or any independently established Application,
Annexure, Affidavit, Vakalatnama, Index, Appendix, Filing Memo or other outer
document; also when court pagination/source identity breaks.

**Exclude:** Petition grounds/prayers, Synopsis and List-of-Dates references,
Applications seeking exemption from filing a certified copy, and background
decisions. These references do not become Impugned Order. Background decisions
filed as exhibits are Annexures.

**Annexure interaction:** A reproduced decision normally remains Annexure. The
directly challenged decision becomes Impugned Order only when challenge/prayer,
court, case, date, decision type and outer Index evidence identify it.

**Repeat:** Separately presented directly challenged decisions may be separate
segments. An explicitly continuous composite order-set remains one segment.

**Current implementation limits:** The non-Index positional/folio fallback
searches within a 40-page window and follows at most 40 pages. Long decisions
therefore require reliable Index/range evidence or future removal of that fixed
window.

### 3.12 Main Petition

**Identity:** The primary substantive pleading for the current Supreme Court
proceeding, including SLP, Writ, Appeal, Transfer, Review, Curative,
Arbitration, Contempt, Original Suit or another supported main proceeding.

**Start:** Its dedicated opening/title and party block, connected to substantive
facts, questions of law, grounds, declarations or prayers. A court caption alone
is insufficient.

**Continue:** Opening pages, party schedule, facts, grounds, questions of law,
prayer, verification, drafting/filing dates and advocate details. Internal
declarations and Annexure references remain part of the petition.

**End:** Before a separately introduced AOR's Certificate, primary supporting
Affidavit or another independently established outer document.

**Exclude:** Incidental I.A.s are Applications. Earlier/lower-court pleadings and
historical Supreme Court pleadings reproduced inside an Annexure remain
enclosed, even when they contain a full petition title and matching parties.

**Repeat:** Only a separately established current primary pleading starts
another Main Petition.

### 3.13 Affidavit

**Identity:** The independent affidavit supporting the current primary
pleading.

**Start:** `AFFIDAVIT` or a clear OCR/letter-spaced variant, supported by sworn
statements plus deponent, verification, oath, notary or attestation evidence,
and current-case purpose.

**Continue:** All sworn paragraphs, verification, deponent signature,
attestation and captionless continuation pages.

**End:** Before the next independently established outer document.

**Exclude:** An Affidavit heading or party names alone. An affidavit expressly
supporting an Application stays with that Application. Earlier-court and
historical Supreme Court affidavits inside Annexures remain enclosed. AOR
certificates/declarations are not affidavits. Missing attestation alone does not
change an otherwise established affidavit's category.

**Repeat:** A separately presented primary-supporting affidavit starts another
document.

### 3.14 Annexures

**Identity:** Each independent outer supporting exhibit of the current paper
book. Runtime labels preserve identifiers such as `Annexure P-1`, `P-2`, etc.

**Start:** An actual exhibit opening established by one or more of:

- an outer Annexure label and separate cover/source opening;
- a matched master-Index/printed-folio range;
- a new source issuer, date, order, notice or letter;
- certification or a clear source transition;
- an outer folio transition consistent with the Index.

An `ANNEXURE P-n` phrase by itself is insufficient.

**Continue:** The complete exhibit, including its cover, translations,
certification, internal pleadings, indexes, affidavits, applications, orders,
appendices and continuation pages.

**End:** Before the next corroborated outer exhibit or independently established
outer document. Never end at a guessed label.

**Exclude:** Annexure references in Index rows, Synopsis, chronology, Petition,
Application, Affidavit, order or Appendix. Local exhibit labels and internal
documents inside the ongoing source record do not create outer boundaries.

**Repeat:** Every genuinely separate exhibit starts a new segment. Preserve the
observed identifier; do not renumber, impose a maximum or infer missing
Annexures from gaps.

**Impugned exception:** The complete directly challenged decision is Impugned
Order when the evidence described in section 3.11 identifies it. Other
decisions remain Annexures.

### 3.15 Application

**Identity:** An independent current-case Application/I.A. for incidental or
procedural relief. Runtime output is numbered `Application 1`, `Application 2`,
and so on according to verified physical order.

**Start:** Its own Application/I.A./Interlocutory Application title or identity,
connected to a body requesting that relief. A caption alone is insufficient.

**Continue:** Pleadings, grounds, prayer, signature and an affidavit expressly
supporting that Application, even when the affidavit has a separate Supreme
Court caption. Keep captionless continuations and internal attachments unless a
real outer exhibit boundary is established.

**End:** Before another independently established current Application or outer
document.

**Exclude:** `PRAYER`, `RESPECTFULLY SHOWETH`, an Annexure citation, continued
heading or application-supporting affidavit does not create another document.
Lower-court and historical Applications reproduced inside Annexures remain
enclosed.

**Repeat:** Adjacent Applications stay separate only when each has its own
supported current-proceeding opening. False model-created Application numbers
on continuation/signature pages are merged back.

### 3.16 Appendix

**Identity:** An independent appendix of the current paper book, commonly
containing statutes, rules, regulations, notifications, schedules or other
reference material.

**Start:** `APPENDIX`, `APPENDIX-I/II`, a numbered/lettered variant or OCR
variant such as `APPENDlX`, corroborated by outer filing context or a matched
Index entry.

**Continue:** Complete reference material, schedules and captionless
continuations.

**End:** Before another independently established Appendix or outer document.

**Exclude:** A case-specific `APPENDIX OF WP(C)` listing exhibits inside a
reproduced judgment; legal text inside an Annexure; an internal Appendix inside
a petition, judgment, Application or record.

**Repeat:** Every independently presented outer Appendix starts a new segment.

### 3.17 Memo of Parties

**Identity:** A separately presented party list adopted as an outer section of
the current paper book.

**Start:** `MEMO OF PARTIES`, `MEMORANDUM OF PARTIES` or a clear variant,
supported by party identities, addresses and procedural designations.

**Continue:** All parties, addresses and continuation pages.

**End:** Before the next independently established outer document.

**Exclude:** Party blocks inside Main Petition, Application or an order. A
Memo inside a reproduced record/Annexure remains enclosed. A lower-court caption
alone does not disqualify an otherwise separately indexed outer party list.

**Repeat:** A distinct independently presented Memo starts another document.

### 3.18 Memo of Appearance

**Identity:** An independent memo recording an advocate/AOR or party's
appearance in the current proceeding.

**Start:** `MEMO OF APPEARANCE`, `MEMORANDUM OF APPEARANCE` or a clear OCR
variant, with an appearance statement and current-case context.

**Continue:** Party details, appearance statement, date, signatures and
captionless integral continuation.

**End:** Before the next independently established outer document, including a
separately starting Vakalatnama.

**Exclude:** Counsel/appearance lists inside a judgment or order; lower-court
and historical Supreme Court appearance memos inside Annexures.

**Same-page rule:** When Memo of Appearance and Vakalatnama share the same
physical page, assign that page to Vakalatnama. If Memo starts on its own page,
keep it separate.

**Repeat:** Each separately starting current-case Memo remains separate.

### 3.19 Vakalatnama

**Identity:** The independent instrument appointing counsel/AOR for a current
party.

**Start:** `VAKALATNAMA`, `VAKALAT NAMA`, `VAKALTNAMA`, `(V/A)` or a clear OCR
variant, supported by appointment/authority language and current-case context.

**Continue:** Party execution, counsel acceptance, stamps, signatures and
captionless integral continuations.

**End:** Before the next independently established outer document.

**Exclude:** An abbreviation, matching parties, advocate signature or court-fee
stamp alone. Lower-court/historical instruments inside Annexure or Application
remain enclosed. PoA/BR is a different authority instrument.

**Repeat:** Every distinct current-proceeding appointment instrument starts a
new segment.

### 3.20 PoA/BR

**Identity:** An independent Power of Attorney or Board Resolution authorizing
a representative to act for a party.

**Start:** Principal/grantor empowerment language or corporate resolution
language such as `RESOLVED THAT`, matched to the represented party and purpose.
A Supreme Court caption is not required.

**Continue:** Integral signatures, schedules, certification and continuation
pages.

**End:** Before the next independently established outer document.

**Exclude:** Generic authority language or advocate signature alone;
Vakalatnama counsel appointment; Memo of Appearance; instruments enclosed in
Annexures or another document.

**Repeat:** A separately presented authority instrument starts another segment.

### 3.21 Filing Memo

**Identity:** An independent submission memo identifying papers tendered in the
current proceeding.

**Start:** `FILING MEMO`, `FILING INDEX`, `INDEX OF FILING` or a clear variant,
supported by submission purpose plus filing advocate/AOR, case particulars or
filing date.

**Continue:** Submission list, continuation sheets, date and signature.

**End:** Before the next independently established outer document.

**Exclude:** Determine function, not title. A paper-book contents table mapping
documents to printed pages is Index. A Filing Memo mentioned as an Index row
remains part of Index. Reproduced lower-court memos remain enclosed.

**Repeat:** A separately presented submission memo starts another document.

### 3.22 Court Fees

**Identity:** An independent court-fee payment, stamp or endorsement sheet for
the current filing.

**Start:** A standalone sheet whose primary function is recording or proving
court fees, supported by payment/stamp/fee-endorsement content and current-case
context.

**Continue:** Payment details and integral receipt/endorsement continuations.

**End:** Before the next independently established outer document.

**Exclude:** Stamps or amounts on Vakalatnama, Application, Cover Page or
another document; Index entries and Checklist answers about fees; lower-court
fee sheets reproduced inside an exhibit.

**Repeat:** A separate outer fee document starts another segment.

### 3.23 Custody Certificate

**Identity:** An independent jail/prison custody or surrender certificate in a
current criminal filing.

**Start:** `CUSTODY CERTIFICATE` or a clear variant, with issuing-authority and
custody, sentence, surrender or release particulars. Match the person and case
where available; a Supreme Court caption is not required.

**Continue:** All custody particulars, continuation sheets and issuing-authority
certification.

**End:** Before the next independently established outer document.

**Exclude:** Custody statements inside a bail/suspension Application and
certificates enclosed in an established Annexure/Application packet.

**Repeat:** A separately presented outer certificate starts another segment.

### 3.24 Rejoinder

**Identity:** A current-party pleading responding to an opposing counter,
reply or written response.

**Start:** A separately presented `REJOINDER` or `REJOINDER AFFIDAVIT` with
response content and current-case context.

**Continue:** Complete response, verification, signatures and integral
attachments.

**End:** Before the next independently established outer document.

**Exclude:** A rejoinder merely mentioned elsewhere; historical/lower-court
rejoinders inside Annexures; a respondent's counter-affidavit or reply.

**Repeat:** Each separately filed response starts another segment. This category
is optional and must never be inferred merely because it would normally exist.

### 3.25 Written Submissions

**Identity:** A separately presented memorandum of legal submissions for the
current proceeding.

**Start:** `WRITTEN SUBMISSIONS`, `WRITTEN ARGUMENTS` or an equivalent heading,
supported by argument text, propositions, authorities or cited cases.

**Continue:** All argument pages, lists of authorities and integral schedules.

**End:** Before the next independently established outer document.

**Exclude:** Synopsis, petition grounds, oral-hearing Record of Proceedings and
submissions quoted in a judgment. Historical submissions inside Annexures
remain enclosed.

**Repeat:** Separately filed sets remain separate. This optional category must
not be inferred as present.

### 3.26 Proof of Service

**Identity:** An independently presented record proving delivery/service of
current-case papers.

**Start:** `PROOF OF SERVICE`, `AFFIDAVIT OF SERVICE` or an acknowledgment
identifying the recipient, served papers and delivery/service details.

**Continue:** Integral service affidavit, receipts, tracking records and
acknowledgments as one packet.

**End:** Before the next independently established outer document.

**Exclude:** Service references or receipts inside Application, Affidavit or
Annexure. Do not infer service from a bare statement.

**Repeat:** Each independent service packet is separate. This category is
optional.

### 3.27 Translation Certificate

**Identity:** An independently presented translator/certifier statement
authenticating a translation in the current filing.

**Start:** `TRANSLATION CERTIFICATE`, `TRANSLATOR'S OATH/DECLARATION` or
equivalent language identifying the source document/language and
translator/certifier.

**Continue:** Certification, oath, signature and integral translator pages.

**End:** Before the next independently established outer document.

**Exclude:** A certificate integral to an Annexure/translation packet when no
independent outer boundary exists; a translation reference; an Application
seeking exemption from translation; historical certificates inside Annexures.

**Repeat:** Every separately presented certificate is separate. This category
is optional.

## 4. Position guidance

Position supports classification but does not replace content evidence. A
common paper-book sequence is:

```text
Cover Page
Index
Advocate's Checklist / Office Report / Listing Proforma / Record of Proceedings
Synopsis
List of Dates & Events
Impugned Order
Main Petition
AOR's Certificate
Affidavit
Annexures / Applications / Appendix / representation documents / other papers
```

Real filings can omit documents, change the order, contain multiple volumes or
place Applications before the challenged decision. The splitter must therefore
use position as corroboration, never as the only reason for a label.

## 5. Boundary review requirements

Mark a segment/page for human review when any of the following remains after
reconciliation:

- the master Index and actual content disagree;
- two categories have equally strong evidence;
- a printed-folio sequence is broken without an explicit next-document start;
- an Annexure identifier appears inside continuous narrative/body text;
- a model boundary lacks an independently supported opening;
- multiple physical regions claim the same non-repeatable outer slot;
- pages remain missing, multiply assigned or outside the source page range;
- OCR quality prevents safe identification.

Do not silently convert uncertainty into a confident boundary.

## 6. Maintaining these rules

When adding or changing a document type:

1. Update the category in `configs/config.json` using the same
   `IDENTITY/START/CONTINUE/END/DO NOT SPLIT/REPEAT` structure.
2. Add or update deterministic heading and nesting logic only when the evidence
   is document-specific and testable.
3. Add positive tests, continuation tests, false-positive tests, enclosing
   Annexure tests and OCR-degraded tests.
4. Add a reviewed ground-truth example to the split evaluation dataset where
   authorization permits.
5. Verify boundary F1, exact range accuracy, page ownership and category
   accuracy before deployment.
6. Keep the canonical name stable unless migrations are provided for persisted
   split artifacts, vector metadata and frontend labels.

