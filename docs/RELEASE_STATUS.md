# Version 1.1.0 release information

[GitHub release](https://github.com/goldriderhszz-hash/root-phenotyping-attunet/releases/tag/v1.1.0) · [Source repository](https://github.com/goldriderhszz-hash/root-phenotyping-attunet)

| Asset | Purpose |
| --- | --- |
| `RootScope-Desktop-Windows-x64.zip` | Ready-to-run Windows x64 application, all five models, offline guides and license notices |
| `RootScope-Desktop-Source.zip` | Application, research and analysis source, models and study records |
| `Additional_file_2_Numerical_records.zip` | Numerical evidence and table/figure replay, without biological images or model graphs |
| `SHA256SUMS.txt` | SHA-256 for the three archives |

The Windows executable is the locally verified version identified in
`validation/v1_1_0_binary_parity.json`. Its one-image source/EXE parity and
five-model self-test apply to that executable hash. Independently rebuilt CI
artifacts may have different binary hashes and should be identified separately.
The original 50-image ONNX validation evaluates the unchanged numerical pipeline;
it does not constitute a new independent biological test.

The historical `research-release` tag and assets remain available as an earlier
version. Use `v1.1.0` for this version. A GitHub version tag supplies a retrievable
software version; no DOI, permanent archive deposit or journal acceptance is claimed.
