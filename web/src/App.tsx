import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import LivePage from "./pages/LivePage";
import ReviewPage from "./pages/ReviewPage";
import MetricsPage from "./pages/MetricsPage";
import DatasetsPage from "./pages/DatasetsPage";
import AuditPage from "./pages/AuditPage";
import SettingsPage from "./pages/SettingsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<LivePage />} />
        <Route path="review" element={<ReviewPage />} />
        <Route path="metrics" element={<MetricsPage />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="audit" element={<AuditPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
