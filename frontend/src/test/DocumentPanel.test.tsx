import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DocumentPanel } from "../components/DocumentPanel";
import { doc } from "./fixtures";

function setup(overrides: Partial<Parameters<typeof DocumentPanel>[0]> = {}) {
  const props = {
    documents: [doc(1)],
    selected: new Set<number>(),
    onToggle: vi.fn(),
    onUpload: vi.fn().mockResolvedValue(undefined),
    onDelete: vi.fn().mockResolvedValue(undefined),
    maxUploadMb: 25,
    ocr: true,
    ...overrides,
  };
  const { container } = render(<DocumentPanel {...props} />);
  return { props, container, user: userEvent.setup() };
}

describe("DocumentPanel", () => {
  it("shows status, chunk and page counts", () => {
    setup({
      documents: [
        doc(1, { filename: "guide.pdf" }),
        doc(2, { filename: "new.txt", status: "processing" }),
        doc(3, { filename: "bad.pdf", status: "failed", error: "Stream has ended unexpectedly" }),
      ],
    });
    expect(screen.getByText(/5 chunks · 3 pages/)).toBeInTheDocument();
    expect(screen.getByText("processing")).toBeInTheDocument();
    expect(screen.getByText("Stream has ended unexpectedly")).toBeInTheDocument();
  });

  it("marks documents that were read with OCR", () => {
    setup({ documents: [doc(1, { ocr_pages: 3 }), doc(2, { ocr_pages: 1 })] });
    expect(screen.getByText("· OCR all pages")).toBeInTheDocument();
    expect(screen.getByText("· OCR 1 page")).toBeInTheDocument();
  });

  it("mentions OCR support only when the server has it", () => {
    setup({ ocr: false });
    expect(screen.queryByText("Scanned PDFs are read with OCR")).not.toBeInTheDocument();
  });

  it("only lets ready documents be selected", async () => {
    const { props, user } = setup({
      documents: [doc(1, { filename: "a.pdf" }), doc(2, { filename: "b.pdf", status: "processing" })],
    });
    expect(screen.getByRole("checkbox", { name: "Search only b.pdf" })).toBeDisabled();
    await user.click(screen.getByRole("checkbox", { name: "Search only a.pdf" }));
    expect(props.onToggle).toHaveBeenCalledWith(1);
  });

  it("describes the current search scope", () => {
    setup({ documents: [doc(1), doc(2)], selected: new Set([1, 2]) });
    expect(screen.getByText("Searching 2 selected documents.")).toBeInTheDocument();
  });

  it("uploads chosen files", async () => {
    const { props, container, user } = setup();
    const files = [new File(["a"], "a.txt", { type: "text/plain" }), new File(["b"], "b.pdf")];
    await user.upload(container.querySelector('input[type="file"]') as HTMLInputElement, files);
    expect(props.onUpload).toHaveBeenCalledWith(files);
  });

  it("deletes a document", async () => {
    const { props, user } = setup();
    await user.click(screen.getByRole("button", { name: "Delete doc1.pdf" }));
    expect(props.onDelete).toHaveBeenCalledWith(doc(1));
  });
});
