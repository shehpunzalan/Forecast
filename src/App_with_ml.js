
import React, { useEffect, useState, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "./App_with_ml.css";

const rnd = (min, max) => Math.floor(Math.random() * (max - min + 1)) + min;

export default function App_TF() {
  const [products, setProducts] = useState([]);
  const [model, setModel] = useState(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [useML, setUseML] = useState(true);
  const [accHistory, setAccHistory] = useState([]);

  const canvasRef = useRef(null);
  const canvasLabelsRef = useRef(null);

  // Fetch product data
  useEffect(() => {
    async function fetchProducts() {
      try {
        const res = await fetch("https://dummyjson.com/products?limit=100");
        const json = await res.json();
        const items = json.products || [];

        const mapped = items.map((p, i) => ({
          id: p.id ?? i,
          name: p.title ?? `Product ${i}`,
          currentInventory: rnd(0, 300),
          avgSalesPerWeek: rnd(1, 80),
          daysToReplenish: rnd(1, 21),
        }));
        setProducts(mapped);
      } catch {
        const fallback = Array.from({ length: 100 }).map((_, i) => ({
          id: i + 1,
          name: `Demo Product ${i + 1}`,
          currentInventory: rnd(0, 300),
          avgSalesPerWeek: rnd(1, 80),
          daysToReplenish: rnd(1, 21),
        }));
        setProducts(fallback);
      }
    }

    fetchProducts();
  }, []);

  // Train the model
  useEffect(() => {
    if (products.length === 0) return;

    async function train() {
      setLoadingModel(true);
      const xs = [];
      const ys = [];
      const trainingSamples = 4000;

      for (let i = 0; i < trainingSamples; i++) {
        const inv = rnd(0, 300);
        const sales = rnd(1, 80);
        const lead = rnd(1, 21);
        const daysOfCover = (inv / sales) * 7;
        const reorder = daysOfCover <= lead ? 1 : 0;
        xs.push([inv, sales, lead]);
        ys.push(reorder);
      }

      const xsTensor = tf.tensor2d(xs);
      const ysTensor = tf.tensor2d(ys, [ys.length, 1]);

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 10, activation: "relu", inputShape: [3] }));
      model.add(tf.layers.dense({ units: 6, activation: "relu" }));
      model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

      model.compile({ optimizer: tf.train.adam(0.01), loss: "binaryCrossentropy", metrics: ["accuracy"] });

      const history = [];

      await model.fit(xsTensor, ysTensor, {
        batchSize: 64,
        epochs: 20,
        shuffle: true,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            history.push(logs.acc);
            setAccHistory([...history]);
          },
        },
      });

      setModel(model);
      setLoadingModel(false);
    }

    train();
  }, [products]);

  // Draw Training Graph
  useEffect(() => {
    const canvas = canvasRef.current;
    const labelCanvas = canvasLabelsRef.current;
    if (!canvas || !labelCanvas) return;

    const ctx = canvas.getContext("2d");
    const labelCtx = labelCanvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);

    if (accHistory.length === 0) return;

    // Draw axis labels
    labelCtx.fillStyle = "#133613ff";
    labelCtx.font = "12px Inter";
    labelCtx.fillText("Accuracy %", 5, 12);
    labelCtx.fillText("100%", 5, 28);
    labelCtx.fillText("50%", 5, canvas.height / 2);
    labelCtx.fillText("0%", 5, canvas.height - 4);

    ctx.strokeStyle = "#207c20ff";
    ctx.lineWidth = 2;
    ctx.beginPath();

    accHistory.forEach((acc, i) => {
      const x = (i / (accHistory.length - 1)) * canvas.width;
      const y = canvas.height - acc * canvas.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.stroke();
  }, [accHistory]);

  function predictReorder(p) {
    if (!useML) {
      const daysOfCover = (p.currentInventory / p.avgSalesPerWeek) * 7;
      return daysOfCover <= p.daysToReplenish;
    }
    if (!model) return false;

    const pred = model
      .predict(tf.tensor2d([[p.currentInventory, p.avgSalesPerWeek, p.daysToReplenish]]))
      .dataSync()[0];

    return pred > 0.5;
  }

  return (
    <div className="app-container">
      <h1>Inventory Forecast</h1>

      <div className="card">
        <label className="toggle-label">
          <input type="checkbox" checked={useML} onChange={(e) => setUseML(e.target.checked)} />
          Use Machine Learning
        </label>

        {loadingModel ? <div>Training model...</div> : <div>✔ Model Ready</div>}

        <div style={{ marginTop: 18 }}>
          <div className="canvas-title">Training Accuracy</div>

          <div className="canvas-box">
            <canvas ref={canvasLabelsRef} width={60} height={150} style={{ float: "left" }} />
            <canvas ref={canvasRef} width={460} height={150} />
          </div>
        </div>
      </div>

      <table className="table">
        <thead>
          <tr>
            <th>Product Name</th>
            <th>Current Inventory Level</th>
            <th>Average Sales per Week</th>
            <th>Days to Replenish the Product</th>
            <th>Suggestion</th>
          </tr>
        </thead>
        <tbody>
          {products.map((p) => (
            <tr key={p.id}>
              <td>{p.name}</td>
              <td>{p.currentInventory}</td>
              <td>{p.avgSalesPerWeek}</td>
              <td>{p.daysToReplenish}</td>
              <td>
                {predictReorder(p) ? (
                  <span className="status-reorder">Reorder</span>
                ) : (
                  <span className="status-ok">No Reorder</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}