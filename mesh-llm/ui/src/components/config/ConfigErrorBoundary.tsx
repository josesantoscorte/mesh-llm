import { Component, type ErrorInfo, type ReactNode, useEffect, useState } from 'react';

import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Button } from '../ui/button';

type Props = {
  children: ReactNode;
};

type State = {
  hasError: boolean;
  error: Error | null;
  showDetails: boolean;
};

/**
 * Hook to catch unhandled Promise rejections and throw them into React's error boundary.
 * Registers a window unhandledrejection listener that uses setState to throw the error.
 */
export function useAsyncError() {
  const [, setError] = useState();

  useEffect(() => {
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      // Throw the error via setState so React's error boundary catches it
      setError(() => {
        throw event.reason;
      });
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  return (error: Error) => {
    setError(() => {
      throw error;
    });
  };
}

class ConfigErrorBoundaryImpl extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, showDetails: false };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('[ConfigErrorBoundary]', error, info.componentStack);
  }

  handleReload = () => {
    window.location.reload();
  };

  toggleDetails = () => {
    this.setState((prev) => ({ showDetails: !prev.showDetails }));
  };

  render() {
    if (this.state.hasError) {
      return (
        <Card className="mx-auto max-w-lg">
          <CardHeader>
            <CardTitle data-testid="error-boundary-message">Something went wrong</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              The configuration panel encountered an unexpected error. You can try reloading.
            </p>

            <div className="flex gap-2">
              <Button size="sm" onClick={this.handleReload}>
                Reload
              </Button>
              <Button variant="ghost" size="sm" onClick={this.toggleDetails}>
                {this.state.showDetails ? 'Hide details' : 'Show details'}
              </Button>
            </div>

            {this.state.showDetails && this.state.error ? (
              <pre className="mt-2 max-h-40 overflow-auto rounded-md border bg-muted/40 p-3 text-[11px] text-muted-foreground">
                {this.state.error.message}
                {this.state.error.stack ? `\n\n${this.state.error.stack}` : ''}
              </pre>
            ) : null}
          </CardContent>
        </Card>
      );
    }

    return this.props.children;
  }
}

/**
 * Wrapper component that combines the class error boundary with async error handling.
 * The hook catches unhandled Promise rejections, and the class component catches sync errors.
 */
export function ConfigErrorBoundary({ children }: Props) {
  useAsyncError();
  return <ConfigErrorBoundaryImpl>{children}</ConfigErrorBoundaryImpl>;
}
