import React, { useState, useEffect } from 'react';

const Search = () => {
  const [query, setQuery] = useState('');
  const [collection, setCollection] = useState('');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [topK, setTopK] = useState(3);
  const [threshold, setThreshold] = useState(0.7);
  const [collections, setCollections] = useState([]);
  const [providers, setProviders] = useState([]);
  const [selectedProvider, setSelectedProvider] = useState('milvus');
  const [wordCountThreshold, setWordCountThreshold] = useState(100);

  // 加载向量数据库providers和collections
  useEffect(() => {
    const fetchData = async () => {
      try {
        // 获取providers列表
        const providersResponse = await fetch('http://localhost:8001/providers');
        const providersData = await providersResponse.json();
        setProviders(providersData.providers);

        // 获取collections列表
        const collectionsResponse = await fetch(`http://localhost:8001/collections?provider=${selectedProvider}`);
        const collectionsData = await collectionsResponse.json();
        setCollections(collectionsData.collections);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [selectedProvider]);

  const handleSearch = async () => {
    if (!query || !collection) return;

    setIsSearching(true);
    try {
      console.log('Search request:', {
        query,
        collection_id: collection,
        top_k: topK,
        threshold,
        word_count_threshold: wordCountThreshold
      });

      const response = await fetch('http://localhost:8001/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          collection_id: collection,
          top_k: topK,
          threshold,
          word_count_threshold: wordCountThreshold
        }),
      });

      const data = await response.json();
      
      console.log('Search response:', data);
      
      setResults(data.results || []);
    } catch (error) {
      console.error('Error searching:', error);
      setResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Similarity Search</h2>
      
      <div className="grid grid-cols-12 gap-6">
        {/* Left Panel - Search Controls */}
        <div className="col-span-3 space-y-4">
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Vector Database</label>
                <select
                  value={selectedProvider}
                  onChange={(e) => setSelectedProvider(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">Choose a provider...</option>
                  {providers && providers.length > 0 ? (
                    providers.map(provider => (
                      <option key={provider.id} value={provider.id}>
                        {provider.name}
                      </option>
                    ))
                  ) : (
                    <option value="" disabled>No providers available</option>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Collection</label>
                <select
                  value={collection}
                  onChange={(e) => setCollection(e.target.value)}
                  className="block w-full p-2 border rounded"
                >
                  <option value="">Choose a collection...</option>
                  {collections && collections.length > 0 ? (
                    collections.map(coll => (
                      <option key={coll.id} value={coll.id}>
                        {coll.name} ({coll.count} documents)
                      </option>
                    ))
                  ) : (
                    <option value="" disabled>No collections available</option>
                  )}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Top K Results</label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  min="1"
                  max="10"
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

              <div>
                <label className="block text-sm font-medium mb-1">
                  Minimum Word Count: {wordCountThreshold}
                </label>
                <input
                  type="range"
                  value={wordCountThreshold}
                  onChange={(e) => setWordCountThreshold(parseInt(e.target.value))}
                  min="0"
                  max="500"
                  step="10"
                  className="block w-full"
                />
              </div>

              <button 
                onClick={handleSearch}
                disabled={isSearching}
                className="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-blue-300"
              >
                {isSearching ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Panel - Query Input and Results */}
        <div className="col-span-9 space-y-4">
          {/* Query Input */}
          <div className="p-4 border rounded-lg bg-white shadow-sm">
            <div>
              <label className="block text-sm font-medium mb-1">Your Question</label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your search query..."
                className="block w-full p-2 border rounded h-32 resize-none"
              />
            </div>
          </div>

          {/* Results Panel */}
          <div className="border rounded-lg bg-white shadow-sm">
            {results.length > 0 ? (
              <div className="p-4">
                <h3 className="text-xl font-semibold mb-4">Search Results</h3>
                <div className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
                  {results.map((result, idx) => (
                    <div key={idx} className="p-4 border rounded bg-gray-50">
                      <div className="flex justify-between items-start mb-2">
                        <span className="font-medium text-sm text-gray-500">
                          Match Score: {(result.score * 100).toFixed(1)}%
                        </span>
                        <div className="text-sm text-gray-500">
                          <div>Source: {result.metadata.source}</div>
                          <div>Page: {result.metadata.page}</div>
                          <div>Chunk: {result.metadata.chunk}</div>
                        </div>
                      </div>
                      <p className="text-sm whitespace-pre-wrap">{result.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500 p-6">
                <p>Enter your search query and click Search to find similar documents.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Search;