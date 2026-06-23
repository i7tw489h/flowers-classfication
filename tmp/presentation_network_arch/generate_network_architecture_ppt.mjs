import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = process.cwd();
const PREVIEW_DIR = path.join(ROOT, "rendered_slides");
const LAYOUT_DIR = path.join(ROOT, "slide_layouts");
const OUTPUT_DIR = path.join(ROOT, "..", "..", "outputs");
const OUTPUT_PPTX = path.join(
  OUTPUT_DIR,
  "deep_learning_network_architecture_cbam.pptx",
);

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addModule(slide, options) {
  const {
    name,
    left,
    top,
    width,
    height,
    text,
    fill,
    lineFill = "#cbd5e1",
    lineWidth = 1.5,
    textColor = "#111827",
    fontSize = 22,
    bold = true,
    radius = "rounded-2xl",
  } = options;

  const shape = slide.shapes.add({
    geometry: "roundRect",
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
    borderRadius: radius,
    shadow: "shadow-none",
  });
  shape.text = text;
  shape.text.style = {
    fontSize,
    bold,
    color: textColor,
    typeface: "Arial",
    alignment: "center",
    verticalAlignment: "middle",
    autoFit: "shrinkText",
    insets: { top: 6, right: 6, bottom: 6, left: 6 },
  };
  return shape;
}

function addTitle(slide, text, top = 42) {
  const title = slide.shapes.add({
    geometry: "textbox",
    name: `title-${text.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
    position: { left: 60, top, width: 1480, height: 58 },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  title.text = text;
  title.text.style = {
    fontSize: 40,
    bold: true,
    color: "#111827",
    typeface: "Arial",
    alignment: "left",
    verticalAlignment: "middle",
    wrap: "none",
  };
  return title;
}

function addArrow(slide, fromShape, toShape, fromSide = "right", toSide = "left") {
  return slide.shapes.connect(fromShape, toShape, {
    kind: "straight",
    fromSide,
    toSide,
    line: { style: "solid", fill: "#475569", width: 2.2 },
    head: { type: "arrow", width: "med", length: "med" },
  });
}

function addElbowArrow(slide, fromShape, toShape, fromSide, toSide) {
  return slide.shapes.connect(fromShape, toShape, {
    kind: "elbow",
    fromSide,
    toSide,
    line: { style: "solid", fill: "#64748b", width: 2 },
    head: { type: "arrow", width: "med", length: "med" },
  });
}

function buildArchitectureSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";

  addTitle(slide, "ResNet-CBAM Network Architecture");

  const top = 388;
  const mainHeight = 102;
  const cbamSize = 74;
  const cbamTop = top + (mainHeight - cbamSize) / 2;
  const gap = 10;

  const palette = {
    input: { fill: "#f3f4f6", line: "#9ca3af", text: "#111827" },
    stem: { fill: "#e5e7eb", line: "#94a3b8", text: "#111827" },
    l1: { fill: "#7aa6ff", line: "#4472c4", text: "#ffffff" },
    l2: { fill: "#5b8ff9", line: "#3563b8", text: "#ffffff" },
    l3: { fill: "#4472c4", line: "#24478f", text: "#ffffff" },
    l4: { fill: "#2f5597", line: "#1d3f78", text: "#ffffff" },
    cbam: { fill: "#f59e0b", line: "#d97706", text: "#ffffff" },
    head1: { fill: "#78c59a", line: "#2e8b57", text: "#ffffff" },
    head2: { fill: "#55b57d", line: "#2f855a", text: "#ffffff" },
    head3: { fill: "#3d9d67", line: "#226c47", text: "#ffffff" },
    head4: { fill: "#2e8557", line: "#1f5c3d", text: "#ffffff" },
  };

  let x = 60;
  const items = [];

  const push = (shape, width) => {
    items.push(shape);
    x += width + gap;
  };

  push(
    addModule(slide, {
      name: "input",
      left: x,
      top,
      width: 112,
      height: mainHeight,
      text: "Input\n224x224x3",
      fill: palette.input.fill,
      lineFill: palette.input.line,
      textColor: palette.input.text,
      fontSize: 24,
    }),
    112,
  );

  push(
    addModule(slide, {
      name: "conv-stem",
      left: x,
      top,
      width: 150,
      height: mainHeight,
      text: "Conv1\n7x7 / s2\nBN + ReLU + MaxPool",
      fill: palette.stem.fill,
      lineFill: palette.stem.line,
      textColor: palette.stem.text,
      fontSize: 21,
    }),
    150,
  );

  push(
    addModule(slide, {
      name: "layer1",
      left: x,
      top,
      width: 112,
      height: mainHeight,
      text: "Layer1\n256 ch",
      fill: palette.l1.fill,
      lineFill: palette.l1.line,
      textColor: palette.l1.text,
      fontSize: 24,
    }),
    112,
  );

  push(
    addModule(slide, {
      name: "cbam1",
      left: x,
      top: cbamTop,
      width: cbamSize,
      height: cbamSize,
      text: "CBAM",
      fill: palette.cbam.fill,
      lineFill: palette.cbam.line,
      textColor: palette.cbam.text,
      fontSize: 18,
      radius: "rounded-xl",
    }),
    cbamSize,
  );

  push(
    addModule(slide, {
      name: "layer2",
      left: x,
      top,
      width: 112,
      height: mainHeight,
      text: "Layer2\n512 ch",
      fill: palette.l2.fill,
      lineFill: palette.l2.line,
      textColor: palette.l2.text,
      fontSize: 24,
    }),
    112,
  );

  push(
    addModule(slide, {
      name: "cbam2",
      left: x,
      top: cbamTop,
      width: cbamSize,
      height: cbamSize,
      text: "CBAM",
      fill: palette.cbam.fill,
      lineFill: palette.cbam.line,
      textColor: palette.cbam.text,
      fontSize: 18,
      radius: "rounded-xl",
    }),
    cbamSize,
  );

  push(
    addModule(slide, {
      name: "layer3",
      left: x,
      top,
      width: 120,
      height: mainHeight,
      text: "Layer3\n1024 ch",
      fill: palette.l3.fill,
      lineFill: palette.l3.line,
      textColor: palette.l3.text,
      fontSize: 24,
    }),
    120,
  );

  push(
    addModule(slide, {
      name: "cbam3",
      left: x,
      top: cbamTop,
      width: cbamSize,
      height: cbamSize,
      text: "CBAM",
      fill: palette.cbam.fill,
      lineFill: palette.cbam.line,
      textColor: palette.cbam.text,
      fontSize: 18,
      radius: "rounded-xl",
    }),
    cbamSize,
  );

  push(
    addModule(slide, {
      name: "layer4",
      left: x,
      top,
      width: 120,
      height: mainHeight,
      text: "Layer4\n2048 ch",
      fill: palette.l4.fill,
      lineFill: palette.l4.line,
      textColor: palette.l4.text,
      fontSize: 24,
    }),
    120,
  );

  push(
    addModule(slide, {
      name: "cbam4",
      left: x,
      top: cbamTop,
      width: cbamSize,
      height: cbamSize,
      text: "CBAM",
      fill: palette.cbam.fill,
      lineFill: palette.cbam.line,
      textColor: palette.cbam.text,
      fontSize: 18,
      radius: "rounded-xl",
    }),
    cbamSize,
  );

  push(
    addModule(slide, {
      name: "gap",
      left: x,
      top,
      width: 92,
      height: mainHeight,
      text: "Global\nAvg Pool",
      fill: palette.head1.fill,
      lineFill: palette.head1.line,
      textColor: palette.head1.text,
      fontSize: 22,
    }),
    92,
  );

  push(
    addModule(slide, {
      name: "dropout",
      left: x,
      top,
      width: 96,
      height: mainHeight,
      text: "Dropout\np = 0.4",
      fill: palette.head2.fill,
      lineFill: palette.head2.line,
      textColor: palette.head2.text,
      fontSize: 22,
    }),
    96,
  );

  push(
    addModule(slide, {
      name: "fc",
      left: x,
      top,
      width: 72,
      height: mainHeight,
      text: "FC",
      fill: palette.head3.fill,
      lineFill: palette.head3.line,
      textColor: palette.head3.text,
      fontSize: 24,
    }),
    72,
  );

  push(
    addModule(slide, {
      name: "output",
      left: x,
      top,
      width: 110,
      height: mainHeight,
      text: "102-Class\nOutput",
      fill: palette.head4.fill,
      lineFill: palette.head4.line,
      textColor: palette.head4.text,
      fontSize: 23,
    }),
    110,
  );

  for (let i = 0; i < items.length - 1; i += 1) {
    addArrow(slide, items[i], items[i + 1]);
  }

  const backboneLabel = slide.shapes.add({
    geometry: "textbox",
    name: "backbone-label",
    position: { left: 345, top: 290, width: 690, height: 34 },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  backboneLabel.text = "Residual stages with CBAM enhancement";
  backboneLabel.text.style = {
    fontSize: 22,
    bold: false,
    color: "#334155",
    typeface: "Arial",
    alignment: "center",
    verticalAlignment: "middle",
    wrap: "none",
  };

  const headLabel = slide.shapes.add({
    geometry: "textbox",
    name: "classifier-label",
    position: { left: 1180, top: 290, width: 300, height: 34 },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  headLabel.text = "Classification head";
  headLabel.text.style = {
    fontSize: 22,
    bold: false,
    color: "#22543d",
    typeface: "Arial",
    alignment: "center",
    verticalAlignment: "middle",
    wrap: "none",
  };
}

function buildCbamSlide(presentation) {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";

  addTitle(slide, "CBAM Internal Structure");

  const input = addModule(slide, {
    name: "cbam-input",
    left: 60,
    top: 382,
    width: 140,
    height: 96,
    text: "Input\nFeature Map",
    fill: "#eef2ff",
    lineFill: "#93c5fd",
    textColor: "#1e3a8a",
    fontSize: 23,
  });

  const output = addModule(slide, {
    name: "cbam-output",
    left: 1400,
    top: 382,
    width: 140,
    height: 96,
    text: "Output\nFeature Map",
    fill: "#ecfdf5",
    lineFill: "#86efac",
    textColor: "#166534",
    fontSize: 23,
  });

  const channelBox = slide.shapes.add({
    geometry: "roundRect",
    name: "channel-attention-box",
    position: { left: 240, top: 210, width: 540, height: 440 },
    fill: "#fff7ed",
    line: { style: "solid", fill: "#fdba74", width: 1.5 },
    borderRadius: "rounded-2xl",
    shadow: "shadow-none",
  });

  const channelTitle = slide.shapes.add({
    geometry: "textbox",
    name: "channel-attention-title",
    position: { left: 270, top: 232, width: 480, height: 38 },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  channelTitle.text = "Channel Attention";
  channelTitle.text.style = {
    fontSize: 28,
    bold: true,
    color: "#9a3412",
    typeface: "Arial",
    alignment: "center",
    verticalAlignment: "middle",
    wrap: "none",
  };

  const spatialBox = slide.shapes.add({
    geometry: "roundRect",
    name: "spatial-attention-box",
    position: { left: 820, top: 210, width: 540, height: 440 },
    fill: "#f0fdf4",
    line: { style: "solid", fill: "#86efac", width: 1.5 },
    borderRadius: "rounded-2xl",
    shadow: "shadow-none",
  });

  const spatialTitle = slide.shapes.add({
    geometry: "textbox",
    name: "spatial-attention-title",
    position: { left: 850, top: 232, width: 480, height: 38 },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  spatialTitle.text = "Spatial Attention";
  spatialTitle.text.style = {
    fontSize: 28,
    bold: true,
    color: "#166534",
    typeface: "Arial",
    alignment: "center",
    verticalAlignment: "middle",
    wrap: "none",
  };

  const avgPoolC = addModule(slide, {
    name: "channel-avgpool",
    left: 280,
    top: 324,
    width: 110,
    height: 58,
    text: "AvgPool",
    fill: "#ffffff",
    lineFill: "#cbd5e1",
    textColor: "#334155",
    fontSize: 20,
    radius: "rounded-xl",
  });
  const maxPoolC = addModule(slide, {
    name: "channel-maxpool",
    left: 280,
    top: 444,
    width: 110,
    height: 58,
    text: "MaxPool",
    fill: "#ffffff",
    lineFill: "#cbd5e1",
    textColor: "#334155",
    fontSize: 20,
    radius: "rounded-xl",
  });
  const mlp = addModule(slide, {
    name: "shared-mlp",
    left: 445,
    top: 385,
    width: 120,
    height: 76,
    text: "Shared\nMLP",
    fill: "#ffedd5",
    lineFill: "#fb923c",
    textColor: "#9a3412",
    fontSize: 21,
  });
  const sigmoidC = addModule(slide, {
    name: "channel-sigmoid",
    left: 610,
    top: 385,
    width: 90,
    height: 76,
    text: "Sigmoid",
    fill: "#fff7ed",
    lineFill: "#fdba74",
    textColor: "#9a3412",
    fontSize: 20,
  });
  const mulC = addModule(slide, {
    name: "channel-multiply",
    left: 700,
    top: 385,
    width: 56,
    height: 76,
    text: "x",
    fill: "#fed7aa",
    lineFill: "#f97316",
    textColor: "#9a3412",
    fontSize: 26,
  });

  const refined = addModule(slide, {
    name: "refined-feature",
    left: 770,
    top: 382,
    width: 90,
    height: 96,
    text: "Refined\nFeature",
    fill: "#eff6ff",
    lineFill: "#93c5fd",
    textColor: "#1e3a8a",
    fontSize: 20,
  });

  const avgPoolS = addModule(slide, {
    name: "spatial-avgpool",
    left: 860,
    top: 324,
    width: 110,
    height: 58,
    text: "AvgPool",
    fill: "#ffffff",
    lineFill: "#cbd5e1",
    textColor: "#334155",
    fontSize: 20,
    radius: "rounded-xl",
  });
  const maxPoolS = addModule(slide, {
    name: "spatial-maxpool",
    left: 860,
    top: 444,
    width: 110,
    height: 58,
    text: "MaxPool",
    fill: "#ffffff",
    lineFill: "#cbd5e1",
    textColor: "#334155",
    fontSize: 20,
    radius: "rounded-xl",
  });
  const concat = addModule(slide, {
    name: "concat",
    left: 1010,
    top: 385,
    width: 88,
    height: 76,
    text: "Concat",
    fill: "#dcfce7",
    lineFill: "#4ade80",
    textColor: "#166534",
    fontSize: 20,
  });
  const conv = addModule(slide, {
    name: "conv7x7",
    left: 1120,
    top: 385,
    width: 108,
    height: 76,
    text: "7x7 Conv",
    fill: "#bbf7d0",
    lineFill: "#22c55e",
    textColor: "#166534",
    fontSize: 20,
  });
  const sigmoidS = addModule(slide, {
    name: "spatial-sigmoid",
    left: 1248,
    top: 385,
    width: 90,
    height: 76,
    text: "Sigmoid",
    fill: "#f0fdf4",
    lineFill: "#86efac",
    textColor: "#166534",
    fontSize: 20,
  });
  const mulS = addModule(slide, {
    name: "spatial-multiply",
    left: 1350,
    top: 385,
    width: 56,
    height: 76,
    text: "x",
    fill: "#86efac",
    lineFill: "#22c55e",
    textColor: "#166534",
    fontSize: 26,
  });

  addElbowArrow(slide, input, avgPoolC, "right", "left");
  addElbowArrow(slide, input, maxPoolC, "right", "left");
  addElbowArrow(slide, avgPoolC, mlp, "right", "left");
  addElbowArrow(slide, maxPoolC, mlp, "right", "left");
  addArrow(slide, mlp, sigmoidC);
  addArrow(slide, sigmoidC, mulC);
  addArrow(slide, mulC, refined);

  addElbowArrow(slide, refined, avgPoolS, "right", "left");
  addElbowArrow(slide, refined, maxPoolS, "right", "left");
  addElbowArrow(slide, avgPoolS, concat, "right", "left");
  addElbowArrow(slide, maxPoolS, concat, "right", "left");
  addArrow(slide, concat, conv);
  addArrow(slide, conv, sigmoidS);
  addArrow(slide, sigmoidS, mulS);
  addArrow(slide, mulS, output);

  const note = slide.shapes.add({
    geometry: "textbox",
    name: "cbam-note",
    position: { left: 250, top: 700, width: 1110, height: 46 },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  note.text =
    "Sequential attention: channel refinement first, then spatial refinement.";
  note.text.style = {
    fontSize: 20,
    bold: false,
    color: "#475569",
    typeface: "Arial",
    alignment: "center",
    verticalAlignment: "middle",
    wrap: "none",
  };
}

async function main() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });
  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  const presentation = Presentation.create({
    slideSize: { width: 1600, height: 900 },
  });

  buildArchitectureSlide(presentation);
  buildCbamSlide(presentation);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(PREVIEW_DIR, `${stem}.png`), png);

    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(
      path.join(LAYOUT_DIR, `${stem}.layout.json`),
      await layout.text(),
    );
  }

  const montage = await presentation.export({
    format: "webp",
    montage: true,
    scale: 1,
  });
  await writeBlob(path.join(PREVIEW_DIR, "deck-montage.webp"), montage);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT_PPTX);

  console.log(`Saved PPTX to ${OUTPUT_PPTX}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
