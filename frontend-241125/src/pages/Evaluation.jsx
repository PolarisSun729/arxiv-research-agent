import React, { useState, useEffect } from 'react';

const Evaluation = () => {
  const [file, setFile] = useState(null);
  const [collection, setCollection] = useState('');
  const [collections, setCollections] = useState([]);
  const [results, setResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.7);

  // 加载collections列表
  useEffect(() => {
    const fetchCollections = async () => {
      try {
        const response = await fetch('http://localhost:8001/collections?provider=milvus');
        const data = await response.json();
        setCollections(data.collections);
      } catch (error) {
        console.error('Error fetching collections:', error);
      }
    };

    fetchCollections();
  }, []);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file || !collection) return;

    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('collection_id', collection);
    formData.append('top_k', topK);
    formData.append('threshold', threshold);

    try {
      const response = await fetch('http://localhost:8001/evaluate', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Error during evaluation:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Evaluation</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Controls */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">CSV File</label>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Collection</label>
                <select
                  value={collection}
                  onChange={(e) => setCollection(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">Choose a collection...</option>
                  {collections.map(coll => (
                    <option key={coll.id} value={coll.id}>
                      {coll.name} ({coll.count} documents)
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Top K Results</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  min="1"
                  max="20"
                  className="block w-full p-2 border rounded"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Similarity Threshold: {threshold}
                </label>
                <input
                  type="range"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  min="0"
                  max="1"
                  step="0.1"
                  className="block w-full"
                />
              </div>

              <button
                onClick={handleSubmit}
                disabled={isProcessing || !file || !collection}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
              >
                {isProcessing ? 'Processing...' : 'Evaluate'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Panel - Results */}
        <div className="col-span-9">
          {results && results.average_scores && (
            <div className="space-y-4">
              {/* Summary Statistics */}
              <div className="p-4 border rounded-lg bg-white shadow-sm">
                <h3 className="text-lg font-semibold mb-3">Summary Statistics</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="p-3 bg-gray-50 rounded">
                    <div className="text-sm text-gray-600">Average Hit Score</div>
                    <div className="text-xl font-bold">
                      {(results.average_scores.score_hit * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 rounded">
                    <div className="text-sm text-gray-600">Average Find Score</div>
                    <div className="text-xl font-bold">
                      {(results.average_scores.score_find * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="p-3 bg-gray-50 rounded">
                    <div className="text-sm text-gray-600">Total Queries</div>
                    <div className="text-xl font-bold">{results.total_queries}</div>
                  </div>
                </div>
              </div>

              {/* Detailed Results */}
              {results.results && results.results.length > 0 && (
                <div className="border rounded-lg bg-white shadow-sm">
                  <div className="p-4">
                    <h3 className="text-lg font-semibold mb-3">Detailed Results</h3>
                    <div className="space-y-4">
                      {results.results.map((result, idx) => (
                        <div key={idx} className="p-4 border rounded bg-gray-50">
                          <div className="flex justify-between items-start mb-2">
                            <div className="font-medium">Query #{idx + 1}</div>
                            <div className="text-sm">
                              <span className="mr-4">Hit Score: {(result.score_hit * 100).toFixed(1)}%</span>
                              <span>Find Score: {(result.score_find * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                          <div className="text-sm mb-2">{result.query}</div>
                          <div className="text-sm text-gray-600">
                            <div>Expected Pages: {result.expected_pages.join(', ')}</div>
                            <div>Found Pages: {result.found_pages.join(', ')}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {isProcessing && (
            <div className="text-center p-4">
              <div className="text-lg text-gray-600">Processing evaluation...</div>
            </div>
          )}

          {!results && !isProcessing && (
            <div className="text-center p-4 border rounded-lg bg-white shadow-sm">
              <div className="text-lg text-gray-600">
                Upload a CSV file and configure the evaluation parameters to begin.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Evaluation; 