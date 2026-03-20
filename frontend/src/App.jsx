import React, { useEffect, useRef, useState } from "react";
import { saveAs } from "file-saver";
import { Document, Packer, Paragraph, TextRun } from "docx";
import { jsPDF } from "jspdf";
import api from "./api";

const processingStages = [
  { key: "uploading", label: "Uploading Video" },
  { key: "moderating", label: "Checking Content Safety" },
  { key: "extracting", label: "Extracting Audio" },
  { key: "transcribing", label: "Transcribing Speech" },
  { key: "summarizing", label: "Generating Summary" },
];

export default function App() {
  const uploadBoxRef = useRef(null);
  const videoUploadRef = useRef(null);
  const videoPlayerRef = useRef(null);
  const progressIntervalRef = useRef(null);

  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("No file selected");
  const [previewURL, setPreviewURL] = useState(null);

  const [showResults, setShowResults] = useState(false);
  const [summaryHTML, setSummaryHTML] = useState(null);
  const [transcriptText, setTranscriptText] = useState("");
  const [transcriptSegments, setTranscriptSegments] = useState([]);
  const [timeKeyPoints, setTimeKeyPoints] = useState([]);
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [questionInput, setQuestionInput] = useState("");
  const [qaAnswer, setQaAnswer] = useState("");
  const [qaSources, setQaSources] = useState([]);
  const [qaError, setQaError] = useState("");
  const [loadingQA, setLoadingQA] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [summaryLength, setSummaryLength] = useState("medium");
  const [summaryStyle, setSummaryStyle] = useState("general");
  const [outputLanguage, setOutputLanguage] = useState("english");
  const [sourceLanguage, setSourceLanguage] = useState("auto");
  const [includeKeyPoints, setIncludeKeyPoints] = useState(true);
  const [loadingSummarize, setLoadingSummarize] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);
  const [progressStep, setProgressStep] = useState("Waiting to start");
  const [progressStage, setProgressStage] = useState("uploading");

  useEffect(() => {
    const videoElement = videoPlayerRef.current;

    return () => {
      if (videoElement && videoElement.src) {
        try {
          URL.revokeObjectURL(videoElement.src);
        } catch (e) {
          console.error("Error revoking object URL:", e);
        }
      }
    };
  }, []);

  useEffect(() => {
    const prev = previewURL;
    return () => {
      if (prev) {
        try {
          URL.revokeObjectURL(prev);
        } catch (e) {
          console.error("Error revoking object URL:", e);
        }
      }
    };
  }, [previewURL]);

  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
    };
  }, []);

  function handleBrowseClick() {
    videoUploadRef.current && videoUploadRef.current.click();
  }

  function onFileChange(e) {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      handleFile(selectedFile);
    }
  }

  function handleFile(selectedFile) {
    if (selectedFile && selectedFile.type && selectedFile.type.startsWith("video/")) {
      if (selectedFile.size === 0) {
        setFile(null);
        setFileName("Selected file is empty. Please choose another video.");
        setErrorMessage("Selected file is empty. Please choose another video.");
        return;
      }

      if (previewURL) {
        try {
          URL.revokeObjectURL(previewURL);
        } catch (e) {
          console.error("Error revoking object URL:", e);
        }
      }

      setFile(selectedFile);
      setFileName(selectedFile.name);
      setErrorMessage("");
      setPreviewURL(URL.createObjectURL(selectedFile));
    } else {
      setFile(null);
      setFileName("Please select a valid video file.");
      setErrorMessage("Please select a valid video file.");
      if (previewURL) {
        try {
          URL.revokeObjectURL(previewURL);
        } catch (e) {
          console.error("Error revoking object URL:", e);
        }
        setPreviewURL(null);
      }
    }
  }

  function onDragOver(e) {
    e.preventDefault();
    uploadBoxRef.current?.classList.add("border-amber-500", "bg-amber-50");
  }

  function onDragLeave() {
    uploadBoxRef.current?.classList.remove("border-amber-500", "bg-amber-50");
  }

  function onDrop(e) {
    e.preventDefault();
    uploadBoxRef.current?.classList.remove("border-amber-500", "bg-amber-50");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  }

  function updateProgressByPercent(nextPercent) {
    const safePercent = Math.max(0, Math.min(99, nextPercent));
    setProgressPercent(safePercent);
    if (safePercent < 20) {
      setProgressStage("uploading");
      setProgressStep("Uploading video");
    } else if (safePercent < 35) {
      setProgressStage("moderating");
      setProgressStep("Checking content safety");
    } else if (safePercent < 55) {
      setProgressStage("extracting");
      setProgressStep("Extracting audio");
    } else if (safePercent < 80) {
      setProgressStage("transcribing");
      setProgressStep("Transcribing speech");
    } else {
      setProgressStage("summarizing");
      setProgressStep("Summarizing content");
    }
  }

  function startProgressSimulation() {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
    }
    progressIntervalRef.current = setInterval(() => {
      setProgressPercent((prev) => {
        let increment = 0.6;
        if (prev < 20) increment = 1.5;
        else if (prev < 35) increment = 1.0;
        else if (prev < 55) increment = 1.1;
        else if (prev < 80) increment = 0.8;
        else if (prev >= 94) increment = 0;

        const next = Math.min(94, prev + increment);
        updateProgressByPercent(next);
        return next;
      });
    }, 400);
  }

  function stopProgressSimulation() {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  }

  function makeSafeBaseName(name) {
    const base = (name || "summary")
      .replace(/\.[^/.]+$/, "")
      .replace(/[<>:"/\\|?*\u0000-\u001F]/g, "_")
      .trim();
    return base || "summary";
  }

  function buildSummaryExportText() {
    const lines = [];
    lines.push("AI Video Summarizer Export");
    lines.push("");
    lines.push("Summary");
    lines.push(summaryHTML || "No summary available.");
    lines.push("");

    if (timeKeyPoints.length > 0) {
      lines.push("Time-Based Key Points");
      timeKeyPoints.forEach((item, idx) => {
        const label = `${item.start_label || "00:00"} - ${item.end_label || "00:00"}`;
        lines.push(`${idx + 1}. [${label}] ${item.point || ""}`);
      });
      lines.push("");
    }

    if (qaAnswer) {
      lines.push("Interactive Q&A");
      lines.push(qaAnswer);
      lines.push("");
    }

    lines.push("Transcript");
    lines.push(transcriptText || "No transcript available.");
    return lines.join("\n");
  }

  function handleDownloadTxt() {
    const baseName = makeSafeBaseName(fileName);
    const blob = new Blob([buildSummaryExportText()], { type: "text/plain;charset=utf-8" });
    saveAs(blob, `${baseName}_summary.txt`);
  }

  function handleDownloadPdf() {
    const baseName = makeSafeBaseName(fileName);
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const marginX = 44;
    const marginY = 52;
    const lineHeight = 18;
    let cursorY = marginY;

    const writeSection = (title, content) => {
      doc.setFont("helvetica", "bold");
      doc.setFontSize(13);
      if (cursorY > pageHeight - marginY) {
        doc.addPage();
        cursorY = marginY;
      }
      doc.text(title, marginX, cursorY);
      cursorY += lineHeight;

      doc.setFont("helvetica", "normal");
      doc.setFontSize(11);
      const wrapped = doc.splitTextToSize(content || "-", pageWidth - marginX * 2);
      wrapped.forEach((line) => {
        if (cursorY > pageHeight - marginY) {
          doc.addPage();
          cursorY = marginY;
        }
        doc.text(line, marginX, cursorY);
        cursorY += 14;
      });
      cursorY += 10;
    };

    writeSection("Summary", summaryHTML || "No summary available.");
    const keyPointsText =
      timeKeyPoints.length > 0
        ? timeKeyPoints
            .map(
              (item, idx) =>
                `${idx + 1}. [${item.start_label || "00:00"} - ${item.end_label || "00:00"}] ${
                  item.point || ""
                }`
            )
            .join("\n")
        : "No key points available.";
    writeSection("Time-Based Key Points", keyPointsText);
    if (qaAnswer) {
      writeSection("Interactive Q&A", qaAnswer);
    }
    writeSection("Transcript", transcriptText || "No transcript available.");

    doc.save(`${baseName}_summary.pdf`);
  }

  async function handleDownloadDocx() {
    const baseName = makeSafeBaseName(fileName);
    const children = [
      new Paragraph({
        children: [new TextRun({ text: "AI Video Summarizer Export", bold: true, size: 30 })],
        spacing: { after: 260 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Summary", bold: true, size: 26 })],
      }),
      new Paragraph(summaryHTML || "No summary available."),
      new Paragraph(""),
      new Paragraph({
        children: [new TextRun({ text: "Time-Based Key Points", bold: true, size: 26 })],
      }),
    ];

    if (timeKeyPoints.length === 0) {
      children.push(new Paragraph("No key points available."));
    } else {
      timeKeyPoints.forEach((item, idx) => {
        children.push(
          new Paragraph(
            `${idx + 1}. [${item.start_label || "00:00"} - ${item.end_label || "00:00"}] ${
              item.point || ""
            }`
          )
        );
      });
    }

    children.push(new Paragraph(""));
    if (qaAnswer) {
      children.push(
        new Paragraph({
          children: [new TextRun({ text: "Interactive Q&A", bold: true, size: 26 })],
        })
      );
      children.push(new Paragraph(qaAnswer));
      children.push(new Paragraph(""));
    }
    children.push(
      new Paragraph({
        children: [new TextRun({ text: "Transcript", bold: true, size: 26 })],
      })
    );
    children.push(new Paragraph(transcriptText || "No transcript available."));

    const doc = new Document({
      sections: [{ properties: {}, children }],
    });

    const blob = await Packer.toBlob(doc);
    saveAs(blob, `${baseName}_summary.docx`);
  }

  const handleSummarizeAPI = async () => {
    if (!file) {
      alert("Please upload a video file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("summary_length", summaryLength);
    formData.append("summary_style", summaryStyle);
    formData.append("transcription_task", outputLanguage === "english" ? "translate" : "transcribe");
    formData.append("source_language", sourceLanguage);
    formData.append("include_key_points", String(includeKeyPoints));

    setShowResults(true);
    setLoadingSummarize(true);
    setErrorMessage("");
    setSummaryHTML(null);
    setTranscriptText("");
    setTranscriptSegments([]);
    setTimeKeyPoints([]);
    setSuggestedQuestions([]);
    setQuestionInput("");
    setQaAnswer("");
    setQaSources([]);
    setQaError("");
    setProgressPercent(5);
    setProgressStage("uploading");
    setProgressStep("Uploading video");
    startProgressSimulation();

    try {
      const res = await api.post("/process-video", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 10 * 60 * 1000,
        onUploadProgress: (event) => {
          if (!event?.total) return;
          const uploadedPercent = Math.round((event.loaded * 100) / event.total);
          const scaledPercent = Math.min(18, 5 + uploadedPercent * 0.13);
          setProgressPercent((prev) => {
            const next = Math.max(prev, scaledPercent);
            updateProgressByPercent(next);
            return next;
          });
        },
      });

      const data = res.data || {};
      setSummaryHTML(data.summary || "No summary returned from server.");
      setTranscriptText(data.transcript_text || "");
      setTranscriptSegments(Array.isArray(data.transcript_segments) ? data.transcript_segments : []);
      setTimeKeyPoints(Array.isArray(data.time_key_points) ? data.time_key_points : []);
      setSuggestedQuestions(Array.isArray(data.suggested_questions) ? data.suggested_questions : []);
      stopProgressSimulation();
      setProgressPercent(100);
      setProgressStage("summarizing");
      setProgressStep("Summary ready");
    } catch (e) {
      stopProgressSimulation();
      const backendPayload = e?.response?.data;
      const status = e?.response?.status;
      const backendDetail = backendPayload?.detail;
      const debugInfo = {
        message: e?.message || "Unknown error",
        code: e?.code || null,
        status: status || null,
        detail: backendDetail || null,
        response: backendPayload || null,
        requestUrl: e?.config?.url || null,
        baseURL: e?.config?.baseURL || null,
      };

      if (typeof backendDetail === "string" && backendDetail.trim()) {
        setErrorMessage(backendDetail);
      } else if (backendDetail) {
        setErrorMessage(`Backend error: ${JSON.stringify(backendDetail)}`);
      } else if (status) {
        setErrorMessage(`Request failed with status ${status}. Check backend logs for details.`);
      } else {
        setErrorMessage("Network/CORS error. Backend may be down or blocked by CORS.");
      }
      console.error("Summarize API error (raw):", e);
      console.error("Summarize API error (debug):", debugInfo);
    } finally {
      setLoadingSummarize(false);
    }
  };

  async function onSummarizeClick() {
    await handleSummarizeAPI();
  }

  async function askVideoQuestion(nextQuestion) {
    const askedQuestion = (nextQuestion ?? questionInput).trim();
    if (!askedQuestion) {
      setQaError("Enter a question about the video.");
      return;
    }

    setQaError("");
    setQaAnswer("");
    setQaSources([]);
    setLoadingQA(true);

    try {
      const res = await api.post("/ask-video", {
        question: askedQuestion,
        transcript_text: transcriptText,
        transcript_segments: transcriptSegments,
      });

      const data = res.data || {};
      setQuestionInput(askedQuestion);
      setQaAnswer(data.answer || "");
      setQaSources(Array.isArray(data.sources) ? data.sources : []);
    } catch (e) {
      const backendDetail = e?.response?.data?.detail;
      setQaError(
        typeof backendDetail === "string" && backendDetail.trim()
          ? backendDetail
          : "Could not answer that question from the transcript."
      );
    } finally {
      setLoadingQA(false);
    }
  }

  function seekVideoTo(seconds) {
    const video = videoPlayerRef.current;
    const time = Number(seconds);

    if (!video || !Number.isFinite(time)) {
      return;
    }

    const hasDuration = Number.isFinite(video.duration) && video.duration > 0;
    const safeTime = hasDuration ? Math.min(Math.max(0, time), Math.max(0, video.duration - 0.25)) : Math.max(0, time);

    video.currentTime = safeTime;
    video.play().catch(() => {});
  }

  return (
    <div className="container mx-auto px-4 py-6 md:px-8 md:py-10 max-w-7xl min-h-screen">
      <header className="text-center mb-8 md:mb-10">
        <h1 className="mt-4 text-3xl md:text-5xl font-bold text-slate-900 tracking-tight" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
          AI Video Summarizer
        </h1>
        <p className="text-slate-700 mt-3 text-sm md:text-base max-w-2xl mx-auto">
          Convert long videos into crisp summaries with multilingual transcription and translation support.
        </p>
      </header>

      <main id="main-content">
        {!showResults && (
          <div
            id="upload-section"
            className="bg-white/95 backdrop-blur-sm p-5 md:p-8 rounded-2xl shadow-lg border border-slate-200"
          >
            <h2 className="text-xl font-semibold mb-4 text-slate-800" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>1. Start Here</h2>
            <div
              id="upload-box"
              ref={uploadBoxRef}
              className="border-2 border-dashed border-slate-300 rounded-xl p-8 text-center cursor-pointer hover:border-amber-500 hover:bg-amber-50 transition-colors"
              onClick={handleBrowseClick}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
            >
              <input
                type="file"
                id="video-upload"
                ref={videoUploadRef}
                className="hidden"
                accept="video/*"
                onChange={onFileChange}
              />
              <div className="flex flex-col items-center">
                <svg
                  className="w-12 h-12 text-slate-400 mb-4"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth="1.5"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                  />
                </svg>
                <p className="font-semibold text-slate-700">Click to browse or drag & drop your video file</p>
                <p id="file-name" className="text-sm text-slate-500 mt-1">{fileName}</p>
              </div>
            </div>

            <div className="mt-6">
              <h3 className="font-semibold text-slate-800 mb-3" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>2. Configure Summary</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
                <div>
                  <label
                    htmlFor="summary-length"
                    className="block text-sm font-medium text-slate-600 mb-1"
                  >
                    Summary Length
                  </label>
                  <select
                    id="summary-length"
                    className="w-full p-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                    value={summaryLength}
                    onChange={(e) => setSummaryLength(e.target.value)}
                  >
                    <option value="short">Short</option>
                    <option value="medium">Medium</option>
                    <option value="long">Detailed</option>
                  </select>
                </div>
                <div>
                  <label
                    htmlFor="summary-style"
                    className="block text-sm font-medium text-slate-600 mb-1"
                  >
                    Summary Style
                  </label>
                  <select
                    id="summary-style"
                    className="w-full p-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                    value={summaryStyle}
                    onChange={(e) => setSummaryStyle(e.target.value)}
                  >
                    <option value="general">General</option>
                    <option value="business">Business-focused</option>
                    <option value="student">Student-focused</option>
                    <option value="casual">Casual</option>
                  </select>
                </div>
                <div>
                  <label
                    htmlFor="output-language"
                    className="block text-sm font-medium text-slate-600 mb-1"
                  >
                    Summary Language
                  </label>
                  <select
                    id="output-language"
                    className="w-full p-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                    value={outputLanguage}
                    onChange={(e) => setOutputLanguage(e.target.value)}
                  >
                    <option value="english">English (translate if needed)</option>
                    <option value="original">Original spoken language</option>
                  </select>
                </div>
                <div>
                  <label
                    htmlFor="source-language"
                    className="block text-sm font-medium text-slate-600 mb-1"
                  >
                    Video Speech Language
                  </label>
                  <select
                    id="source-language"
                    className="w-full p-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-amber-500"
                    value={sourceLanguage}
                    onChange={(e) => setSourceLanguage(e.target.value)}
                  >
                    <option value="auto">Auto detect</option>
                    <option value="en">English</option>
                    <option value="hi">Hindi</option>
                    <option value="bn">Bengali</option>
                    <option value="ta">Tamil</option>
                    <option value="te">Telugu</option>
                    <option value="mr">Marathi</option>
                    <option value="gu">Gujarati</option>
                    <option value="kn">Kannada</option>
                    <option value="ml">Malayalam</option>
                    <option value="pa">Punjabi</option>
                    <option value="ur">Urdu</option>
                    <option value="or">Odia</option>
                    <option value="as">Assamese</option>
                  </select>
                </div>
              </div>
              <label className="mt-4 inline-flex items-center gap-2 text-sm text-slate-700">
                <input
                  type="checkbox"
                  className="h-4 w-4 rounded border-slate-300 text-amber-500 focus:ring-amber-500"
                  checked={includeKeyPoints}
                  onChange={(e) => setIncludeKeyPoints(e.target.checked)}
                />
                Generate time-based key points (slightly slower)
              </label>
            </div>

            <div className="mt-6 text-center md:text-right">
              <button
                id="summarize-btn"
                className={`bg-slate-900 text-white font-bold py-3 px-8 rounded-lg hover:bg-amber-500 hover:text-slate-950 transition-all shadow-md ${
                  !file ? "disabled:bg-slate-400 disabled:text-white disabled:cursor-not-allowed" : ""
                }`}
                disabled={!file}
                onClick={onSummarizeClick}
              >
                Summarize Video
              </button>
            </div>
          </div>
        )}

        {showResults && (
          <div id="results-section" className="mt-8">
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 md:gap-8">
              <div className="xl:col-span-1 bg-white/95 backdrop-blur-sm p-4 md:p-5 rounded-2xl shadow-lg border border-slate-200 h-fit">
                <h2 className="text-xl font-semibold mb-4 text-slate-800" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>Video Preview</h2>
                <video
                  id="video-player"
                  className="w-full rounded-lg"
                  controls
                  ref={videoPlayerRef}
                  src={previewURL || undefined}
                />
              </div>

              <div className="xl:col-span-2 bg-white/95 backdrop-blur-sm p-5 md:p-6 rounded-2xl shadow-lg border border-slate-200">
                <div className="mb-4">
                  <h2 className="text-xl font-semibold text-slate-800" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>Summary</h2>
                </div>
                {errorMessage && (
                  <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-3 text-red-700">
                    {errorMessage}
                  </div>
                )}

                {loadingSummarize ? (
                  <div className="py-8">
                    <p className="font-semibold text-slate-800 mb-3">{progressStep}</p>
                    <div className="h-3 w-full rounded-full bg-slate-200 overflow-hidden">
                      <div
                        className="progress-fill h-full rounded-full transition-all duration-500"
                        style={{ width: `${progressPercent}%` }}
                      ></div>
                    </div>
                    <div className="mt-2 flex justify-between text-xs text-slate-600">
                      <span>Processing</span>
                      <span>{Math.round(progressPercent)}%</span>
                    </div>

                    <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {processingStages.map((stage) => {
                        const stageOrder = processingStages.findIndex((item) => item.key === stage.key);
                        const activeOrder = processingStages.findIndex((item) => item.key === progressStage);
                        const isDone = stageOrder < activeOrder;
                        const isActive = stage.key === progressStage;

                        return (
                          <div
                            key={stage.key}
                            className={`rounded-xl border px-3 py-2 text-sm ${
                              isActive
                                ? "border-amber-400 bg-amber-50 text-amber-900"
                                : isDone
                                  ? "border-emerald-300 bg-emerald-50 text-emerald-900"
                                  : "border-slate-200 bg-slate-50 text-slate-600"
                            }`}
                          >
                            {stage.label}
                          </div>
                        );
                      })}
                    </div>
                    <p className="text-sm text-slate-500 mt-4">
                      Keep this tab open while your video is being processed.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="flex flex-wrap gap-3">
                      <button
                        type="button"
                        onClick={handleDownloadTxt}
                        className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:border-slate-400 hover:bg-slate-50"
                        disabled={!summaryHTML}
                      >
                        Download TXT
                      </button>
                      <button
                        type="button"
                        onClick={handleDownloadPdf}
                        className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:border-slate-400 hover:bg-slate-50"
                        disabled={!summaryHTML}
                      >
                        Download PDF
                      </button>
                      <button
                        type="button"
                        onClick={handleDownloadDocx}
                        className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 hover:border-slate-400 hover:bg-slate-50"
                        disabled={!summaryHTML}
                      >
                        Download DOCX
                      </button>
                    </div>

                    <div
                      id="content-text"
                      className="space-y-4 custom-scrollbar"
                      style={{ maxHeight: 260, overflowY: "auto" }}
                    >
                      <p className="text-slate-600 leading-relaxed whitespace-pre-line">
                        {summaryHTML ||
                          "Your paragraph summary will appear here. It will be a concise overview of the video's content."}
                      </p>
                    </div>

                    {summaryHTML && (
                      <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 md:p-5">
                        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                          <div>
                            <h3 className="text-lg font-semibold text-slate-800" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                              Ask About This Video
                            </h3>
                            <p className="text-sm text-slate-600">
                              Would you like to know more about this topic?
                            </p>
                          </div>
                          <p className="text-xs font-medium text-slate-500">
                            Fixed answer length: 100-150 words
                          </p>
                        </div>

                        {suggestedQuestions.length > 0 && (
                          <div className="mt-4 flex flex-wrap gap-2">
                            {suggestedQuestions.map((question) => (
                              <button
                                key={question}
                                type="button"
                                onClick={() => askVideoQuestion(question)}
                                className="rounded-full border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 hover:border-amber-400 hover:bg-amber-50 hover:text-slate-900"
                              >
                                {question}
                              </button>
                            ))}
                          </div>
                        )}

                        <div className="mt-4 flex flex-col gap-3 md:flex-row">
                          <input
                            type="text"
                            value={questionInput}
                            onChange={(e) => setQuestionInput(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                e.preventDefault();
                                askVideoQuestion();
                              }
                            }}
                            placeholder="Ask something based on the transcript"
                            className="w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 outline-none focus:border-amber-500"
                          />
                          <button
                            type="button"
                            onClick={() => askVideoQuestion()}
                            disabled={loadingQA || !transcriptSegments.length}
                            className="rounded-xl bg-slate-900 px-5 py-3 text-sm font-semibold text-white hover:bg-amber-500 hover:text-slate-950 disabled:cursor-not-allowed disabled:bg-slate-400"
                          >
                            {loadingQA ? "Thinking..." : "Ask"}
                          </button>
                        </div>

                        {qaError && (
                          <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                            {qaError}
                          </div>
                        )}

                        {qaAnswer && (
                          <div className="mt-4 space-y-4">
                            <div className="rounded-xl border border-amber-200 bg-white p-4">
                              <p className="text-xs font-semibold uppercase tracking-wide text-amber-700">
                                Answer
                              </p>
                              <p className="mt-2 text-sm leading-7 text-slate-700 whitespace-pre-line">
                                {qaAnswer}
                              </p>
                            </div>

                            {qaSources.length > 0 && (
                              <div>
                                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                                  Transcript Sources
                                </p>
                                <div className="space-y-2">
                                  {qaSources.map((source, idx) => (
                                    <div key={`${source.start}-${idx}`} className="rounded-xl border border-slate-200 bg-white p-3">
                                      <button
                                        type="button"
                                        onClick={() => seekVideoTo(source.start)}
                                        className="rounded border border-slate-200 bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-600 hover:bg-slate-200"
                                      >
                                        {source.start_label || "00:00"} - {source.end_label || "00:00"}
                                      </button>
                                      <p className="mt-2 text-sm text-slate-600">
                                        {source.text}
                                      </p>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    {includeKeyPoints && (
                      <div>
                        <h3 className="text-lg font-semibold text-slate-800 mb-3" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                          Time-Based Key Points
                        </h3>
                        <div className="space-y-2">
                          {timeKeyPoints.length > 0 ? (
                            timeKeyPoints.map((item, idx) => (
                              <div key={`${item.start}-${idx}`} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                                <p className="text-xs font-semibold text-amber-700 mb-1">
                                  <button
                                    type="button"
                                    onClick={() => seekVideoTo(item.start)}
                                    className="rounded border border-amber-200 bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-800 hover:bg-amber-200"
                                    title="Jump video to this point"
                                  >
                                    {item.start_label || "00:00"} - {item.end_label || "00:00"}
                                  </button>
                                </p>
                                <p className="text-sm text-slate-700">{item.point}</p>
                              </div>
                            ))
                          ) : (
                            <p className="text-sm text-slate-500">Key points will appear after processing.</p>
                          )}
                        </div>
                      </div>
                    )}

                    <div>
                      <h3 className="text-lg font-semibold text-slate-800 mb-3" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                        Full Transcript
                      </h3>
                      <div className="max-h-72 overflow-y-auto custom-scrollbar space-y-2 pr-2">
                        {transcriptSegments.length > 0 ? (
                          transcriptSegments.map((segment, idx) => (
                            <div key={`${segment.start}-${idx}`} className="rounded-lg border border-slate-200 bg-white p-3">
                              <button
                                type="button"
                                onClick={() => seekVideoTo(segment.start)}
                                className="mb-1 rounded border border-slate-200 bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-600 hover:bg-slate-200"
                                title="Jump video to this timestamp"
                              >
                                {segment.start_label || "00:00"} - {segment.end_label || "00:00"}
                              </button>
                              <p className="text-sm text-slate-700">{segment.text}</p>
                            </div>
                          ))
                        ) : (
                          <p className="text-sm text-slate-500">
                            Transcript will appear here after processing.
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
