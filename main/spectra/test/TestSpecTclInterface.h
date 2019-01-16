#ifndef TESTSPECTCLINTERFACE_H
#define TESTSPECTCLINTERFACE_H

#include "SpecTclInterface.h"

class QString;
class TVirtualPad;

namespace Viewer
{

class HistogramList;
class MasterGateList;
class GGate;
class GSlice;
class QRootCanvas;
class HistogramBundle;

class TestSpecTclInterface : public SpecTclInterface
{
private:
    HistogramList* m_pHistList;
    MasterGateList* m_pGateList;

public:
    TestSpecTclInterface();

    virtual ~TestSpecTclInterface();

    virtual void addGate(const GSlice& slice);
    virtual void editGate(const GSlice& slice);
    virtual void deleteGate(const GSlice& slice);

    virtual void addGate(const GGate& slice);
    virtual void editGate(const GGate& slice);
    virtual void deleteGate(const GGate& slice);

    virtual void deleteGate(const QString& name);
    virtual void addOrGate(
      const std::string& name, const std::vector<std::string>& components
    ) {}
    virtual void editOrGate(
      const std::string& name, const std::vector<std::string>& components
    ) {}
    virtual void editAndGate(
      const std::string& name, const std::vector<std::string>& components
    ) {}
    virtual void addAndGate(
      const std::string& name, const std::vector<std::string>& components
    ) {}   
    virtual void enableGatePolling(bool enable);
    virtual bool gatePollingEnabled() const;

    virtual MasterGateList* getGateList();

    virtual void enableHistogramInfoPolling(bool enable);
    virtual bool histogramInfoPollingEnabled() const;

    virtual HistogramList* getHistogramList();

    virtual void requestHistContentUpdate(QRootCanvas* pCanvas);
    virtual void requestHistContentUpdate(TVirtualPad *pPad);
    virtual void requestHistContentUpdate(const QString& hName);

    virtual void clearSpectrum(QRootCanvas* pCanvas) {}
    virtual void clearSpectrum(QString* pName) {}
    virtual void clearAllSpectra() {}
// signals:
    void gateListChanged() {}
    void histogramListChanged() {}
    void histogramContentUpdated(HistogramBundle* pBundle) {}
};

} // end Viewer namespace

#endif // TESTSPECTCLINTERFACE_H
