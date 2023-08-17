<h1 align="center">Rate That ASR (RTASR)</h1>

<div align="center">
	<a  href="https://pypi.org/project/rtasr" target="_blank">
		<img src="https://img.shields.io/pypi/v/rtasr.svg" />
	</a>
	<a  href="https://pypi.org/project/rtasr" target="_blank">
		<img src="https://img.shields.io/pypi/pyversions/rtasr" />
	</a>
	<a  href="https://github.com/Wordcab/rtasr/blob/main/LICENSE" target="_blank">
		<img src="https://img.shields.io/pypi/l/rtasr" />
	</a>
	<a  href="https://github.com/Wordcab/rtasr/actions?workflow=ci-cd" target="_blank">
		<img src="https://github.com/Wordcab/rtasr/workflows/ci-cd/badge.svg" />
	</a>
	<a  href="https://github.com/pypa/hatch" target="_blank">
		<img src="https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg" />
	</a>
</div>

<p align="center"><em>🏆 Run benchmarks against the most common ASR tools on the market.</em></p>

---

## Installation

```bash
git clone https://github.com/Wordcab/rtasr
cd rtasr

pip install .
```

## Commands

The CLI is available through the `rtasr` command.

```bash
rtasr --help
```

### Datasets download

Available datasets are:

* `ami`: [AMI Corpus](http://groups.inf.ed.ac.uk/ami/corpus/)
* `voxconverse`: [VoxConverse](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)

```bash
rtasr download -d <dataset>
```

## Contributing

Be sure to have [hatch](https://hatch.pypa.io/latest/install/) installed.

### Quality

* Run quality checks: `hatch run quality:check`
* Run quality formatting: `hatch run quality:format`

### Testing

* Run tests: `hatch run tests:run`
