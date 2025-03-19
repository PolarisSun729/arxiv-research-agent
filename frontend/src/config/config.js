const config = {
  development: {
    apiBaseUrl: 'http://192.168.172.128:8001'
  },
  production: {
    apiBaseUrl: process.env.REACT_APP_API_BASE_URL || 'http://api.example.com'
  },
  test: {
    apiBaseUrl: 'http://localhost:8001'
  }
};

const env = process.env.NODE_ENV || 'development';
export const apiBaseUrl = config[env].apiBaseUrl;

export default config[env]; 