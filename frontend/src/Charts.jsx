import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const tooltipStyle = {
  background: "var(--tooltip-bg)",
  border: "1px solid var(--tooltip-border)",
  color: "var(--tooltip-text)",
};

export function Chart({ data }) {
  const series = data?.length
    ? Object.keys(data[0]).filter((key) => key !== "date")
    : [];
  const colors = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff"];

  return (
    <div
      className="chart"
      role="img"
      aria-label={`Line chart comparing ${series.join(", ")} over time`}
    >
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis dataKey="date" hide />
          <YAxis stroke="var(--chart-axis)" />
          <Tooltip contentStyle={tooltipStyle} />
          <Legend />
          {series.map((key, index) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={colors[index % colors.length]}
              strokeWidth={2}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function FeatureChart({ data }) {
  return (
    <div
      className="chart"
      role="img"
      aria-label="Model feature importance bar chart"
    >
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis
            dataKey="feature"
            angle={-30}
            textAnchor="end"
            height={70}
            stroke="var(--chart-axis)"
          />
          <YAxis stroke="var(--chart-axis)" />
          <Tooltip contentStyle={tooltipStyle} />
          <Bar dataKey="value" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
