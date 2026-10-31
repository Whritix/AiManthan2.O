import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { LiveProvider } from "./hooks/useLive";
import "./styles.css";

const container = document.getElementById("root");
if (!container) throw new Error("Root element missing");

createRoot(container).render(
  <StrictMode>
    <BrowserRouter>
      <LiveProvider>
        <App />
      </LiveProvider>
    </BrowserRouter>
  </StrictMode>,
);
